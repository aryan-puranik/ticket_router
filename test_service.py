"""
Integration tests against a REAL PostgreSQL (uses a separate database 'ticket_router_test').
Run:  pytest -q test_service.py        (skipped automatically if PostgreSQL is not reachable)
"""
import random
import threading
import time

import psycopg2
import pytest

import config
from db import Database
from models import (ADMIN, ASSIGNED, CLOSED, IN_PROGRESS, L1, L2, NEW, PENDING, RESOLVED, SLA_MS, SUPERVISOR,
                    CapacityError, ForbiddenError, InvalidTransitionError, NotFoundError, ValidationError)
from service import TicketRouter

ROOT = 1          # the default Admin that load() creates in an empty database


@pytest.fixture()
def svc():
    db = Database(dict(config.DB, dbname="ticket_router_test"))
    try:
        db.bootstrap()
        db.init_schema()
    except psycopg2.OperationalError as e:
        pytest.skip(f"PostgreSQL not reachable: {e}")
    db.reset()
    s = TicketRouter(db, btree_order=8)         # small order -> many splits, stricter test
    s.load()
    assert s.list_agents()[0]["role"] == ADMIN and s.list_agents()[0]["id"] == ROOT
    yield s
    db.close()


def mk(s, subject="Cannot login", customer="a@x.com", priority=3, category="login", desc=""):
    return s.create_ticket(subject, desc, customer, category, priority)


def agent(s, name, capacity=5, role=L1, skills=()):
    return s.create_agent(name, capacity, role, list(skills), ROOT)


def check_all(s):
    for tree in (s._by_id, s._by_customer, s._by_time, s._by_status):
        tree.check_invariants()
    for heap in [s._queue, s._sla, *s._cat_queue.values(), *s._pools.values()]:
        heap.check_invariants()
    assert len(s._by_id) == len(s._by_status) == sum(s._status_count.values())
    assert len(s._queue) == s._status_count[NEW] == sum(len(h) for h in s._cat_queue.values())


# ============================================================ queue order, dispatch, lifecycle
def test_priority_order_and_auto_dispatch(svc):
    p3, p1, p4, p2 = (mk(svc, priority=p)["id"] for p in (3, 1, 4, 2))
    assert [t["id"] for t in svc.queue_view()] == [p1, p2, p3, p4]
    ag = agent(svc, "Asha", 1)                                      # joins -> takes the most urgent ticket
    assert svc.get_ticket(p1)["status"] == ASSIGNED and svc.get_ticket(p1)["assignee"] == ag["id"]
    svc.transition(p1, IN_PROGRESS, ag["id"])
    svc.transition(p1, RESOLVED, ag["id"])                          # slot frees -> next most urgent gets it
    assert svc.get_ticket(p2)["status"] == ASSIGNED
    assert [t["id"] for t in svc.queue_view()] == [p3, p4]
    check_all(svc)


def test_same_priority_is_fifo_by_deadline(svc):
    ids = [mk(svc, priority=2)["id"] for _ in range(5)]
    assert [t["id"] for t in svc.queue_view()] == ids


def test_lifecycle_rules(svc):
    a = agent(svc, "A")["id"]
    t = mk(svc)["id"]
    assert svc.get_ticket(t)["status"] == ASSIGNED
    with pytest.raises(InvalidTransitionError):
        svc.transition(t, RESOLVED, a)                              # must go through IN_PROGRESS
    with pytest.raises(ValidationError):
        svc.transition(t, ASSIGNED, a)
    svc.transition(t, IN_PROGRESS, a); svc.transition(t, PENDING, a)
    svc.transition(t, IN_PROGRESS, a); svc.transition(t, RESOLVED, a)
    svc.transition(t, CLOSED, a)
    with pytest.raises(InvalidTransitionError):
        svc.transition(t, NEW, ROOT)                                # CLOSED is terminal
    with pytest.raises(NotFoundError):
        svc.transition(999999, CLOSED, ROOT)


def test_reopen_requeues_and_redispatches(svc):
    a = agent(svc, "A", 1)["id"]
    t = mk(svc)["id"]
    svc.transition(t, IN_PROGRESS, a); svc.transition(t, RESOLVED, a)
    svc.transition(t, NEW, a)                                       # the resolver may reopen
    assert svc.get_ticket(t)["status"] == ASSIGNED                  # agent is free -> picked up immediately
    kinds = [e["action"] for e in svc.audit_trail(t)]
    assert "REOPENED" in kinds and kinds[0] == "CREATED"


def test_least_loaded_agent_balancing(svc):
    agent(svc, "A", 10), agent(svc, "B", 10)
    for _ in range(8):
        mk(svc)
    assert sorted(x["load"] for x in svc.list_agents() if x["takes_tickets"]) == [4, 4]


def test_capacity_respected_and_manual_assign(svc):
    a = agent(svc, "A", 1)
    t1, t2 = mk(svc)["id"], mk(svc)["id"]
    assert svc.get_ticket(t2)["status"] == NEW                      # only 1 slot
    with pytest.raises(CapacityError):
        svc.assign(t2, a["id"], ROOT)
    b = agent(svc, "B", 1)                                          # new agent absorbs the waiting ticket
    assert svc.get_ticket(t2)["assignee"] == b["id"]
    svc.update_agent(b["id"], ROOT, capacity=2)
    t3 = mk(svc)["id"]
    assert svc.get_ticket(t3)["assignee"] == b["id"]
    with pytest.raises(ValidationError):
        svc.assign(t3, ROOT, ROOT)                                  # an Admin does not take tickets
    check_all(svc)


def test_agent_offline_releases_unstarted_tickets(svc):
    a = agent(svc, "A")
    t1, t2 = mk(svc)["id"], mk(svc)["id"]
    svc.transition(t1, IN_PROGRESS, a["id"])
    svc.update_agent(a["id"], a["id"], online=False)                # an agent may set their own availability
    assert svc.get_ticket(t1)["status"] == IN_PROGRESS              # started work stays with the agent
    assert svc.get_ticket(t2)["status"] == NEW and svc.get_ticket(t2)["assignee"] is None
    assert svc.stats()["agents_available"] == 0
    b = agent(svc, "B")
    assert svc.get_ticket(t2)["assignee"] == b["id"]


def test_released_ticket_does_not_bounce_back(svc):
    a = agent(svc, "A", 2)["id"]
    t = mk(svc)["id"]
    svc.transition(t, NEW, a)                                       # A hands it back
    assert svc.get_ticket(t)["status"] == NEW                       # not re-assigned to A
    assert svc.dispatch() == [] and svc.get_ticket(t)["status"] == NEW   # the sweep respects it too
    t2 = mk(svc)["id"]
    assert svc.get_ticket(t2)["assignee"] == a                      # A still gets other work
    b = agent(svc, "B", 1)["id"]
    assert svc.get_ticket(t)["assignee"] == b
    check_all(svc)


def test_priority_change_rekeys_queue(svc):
    t = [mk(svc, priority=3)["id"] for _ in range(3)]
    svc.change_priority(t[2], 1, ROOT, reason="VIP")
    assert [x["id"] for x in svc.queue_view()][0] == t[2]
    with pytest.raises(ValidationError):
        svc.change_priority(t[2], 1, ROOT)
    svc.transition(t[0], CLOSED, ROOT)
    with pytest.raises(InvalidTransitionError):
        svc.change_priority(t[0], 2, ROOT)
    assert t[0] not in [x["id"] for x in svc.queue_view()]
    check_all(svc)


# ============================================================ SLA
def test_sla_escalation_and_breach_is_logged_once(svc):
    t_low, t_hi = mk(svc, priority=4)["id"], mk(svc, priority=1)["id"]
    assert svc.escalate_breached() == []                            # nothing overdue yet
    now = int(time.time() * 1000)
    out = svc.escalate_breached(at_ms=now + SLA_MS[1] + 60_000)     # 1h01m later: only P1 is overdue
    assert [o["ticket_id"] for o in out] == [t_hi] and out[0]["action"] == "BREACHED_P1"
    assert svc.get_ticket(t_hi)["sla_breached"] is True and svc.get_ticket(t_low)["priority"] == 4
    # a breached P1 is not re-flagged every hour (v1 did that and flooded the audit table)
    assert svc.escalate_breached(at_ms=now + 10 * SLA_MS[1]) == []
    svc.escalate_breached(at_ms=now + SLA_MS[4] + 60_000)
    assert svc.get_ticket(t_low)["priority"] == 3                   # escalated one level
    assert [e["action"] for e in svc.audit_trail(t_hi)].count("SLA_BREACHED") == 1
    s2 = TicketRouter(svc.db, btree_order=8); s2.load()             # restart: still not re-flagged
    assert t_hi not in s2._sla and s2.escalate_breached(at_ms=now + SLA_MS[1] + 120_000) == []
    svc.change_priority(t_hi, 2, ROOT)                              # a human resets the clock
    assert svc.get_ticket(t_hi)["sla_breached"] is False and t_hi in svc._sla


def test_escalation_is_batched(svc):
    svc.create_tickets_bulk([{"subject": f"Issue {i}", "customer": "c@x.com", "category": "bug", "priority": 3}
                             for i in range(1000)])
    svc.create_tickets_bulk([{"subject": f"More {i}", "customer": "c@x.com", "category": "bug", "priority": 3}
                             for i in range(200)])
    before = svc.db.commits
    out = svc.escalate_breached(at_ms=int(time.time() * 1000) + SLA_MS[3] + 1000, batch=500)
    assert len(out) == 1200
    assert svc.db.commits - before == 3                             # 1200 changes -> 3 commits, not 1200
    assert svc.verify_all_audits()["valid"]
    check_all(svc)


def test_pending_pauses_sla(svc):
    a = agent(svc, "A")["id"]
    t = mk(svc, priority=1)["id"]
    svc.transition(t, IN_PROGRESS, a); svc.transition(t, PENDING, a)
    assert svc.stats()["sla_tracked"] == 0
    svc.transition(t, IN_PROGRESS, a)
    assert svc.stats()["sla_tracked"] == 1


# ============================================================ roles: permissions
def test_permissions_by_role(svc):
    l1 = agent(svc, "Lina", 5, L1)["id"]
    l2 = agent(svc, "Sam", 5, L2)["id"]
    sup = agent(svc, "Sue", 1, SUPERVISOR)["id"]
    t_l1 = mk(svc, priority=3)["id"]                                # P3 prefers L1
    t_l2 = mk(svc, priority=1)["id"]                                # P1 prefers L2
    assert svc.get_ticket(t_l1)["assignee"] == l1 and svc.get_ticket(t_l2)["assignee"] == l2

    with pytest.raises(ForbiddenError):
        svc.transition(t_l2, IN_PROGRESS, l1)                       # not their ticket
    with pytest.raises(ForbiddenError):
        svc.transition(t_l1, IN_PROGRESS, None)                     # nobody signed in
    with pytest.raises(ForbiddenError):
        svc.transition(t_l1, IN_PROGRESS, 9999)                     # unknown staff id
    svc.transition(t_l1, IN_PROGRESS, l1)                           # own ticket: fine
    with pytest.raises(ForbiddenError):
        svc.change_priority(t_l1, 2, l1)                            # L1 may not re-prioritise
    svc.change_priority(t_l2, 2, l2)                                # L2 may, on own tickets
    with pytest.raises(ForbiddenError):
        svc.change_priority(t_l1, 2, l2)                            # ... but not on someone else's
    with pytest.raises(ForbiddenError):
        svc.assign(t_l1, l2, l2)                                    # agents cannot re-assign
    svc.assign(t_l1, l2, sup)                                       # a Supervisor can
    assert svc.get_ticket(t_l1)["assignee"] == l2
    svc.transition(t_l1, RESOLVED, sup)                             # a Supervisor may move any ticket
    with pytest.raises(ForbiddenError):
        svc.create_agent("X", 5, L1, [], sup)            # only an Admin manages the roster
    with pytest.raises(ForbiddenError):
        svc.update_agent(l1, sup, capacity=9)
    with pytest.raises(ForbiddenError):
        svc.update_agent(l2, l1, online=False)                      # not someone else's availability
    svc.update_agent(l1, sup, online=False)                         # Supervisor may
    with pytest.raises(ForbiddenError):
        svc.check(l2, "system_operations")
    svc.check(sup, "system_operations")
    last = svc.audit_trail(t_l1)[-1]
    assert last["actor"] == "Sue" and '"actor_role":"SUPERVISOR"' in last["detail"]


def test_roster_rules(svc):
    with pytest.raises(ValidationError):
        svc.update_agent(ROOT, ROOT, role=L1)                       # never lose the last Admin
    with pytest.raises(ValidationError):
        agent(svc, "Bad", 5, L1, ["cooking"])                       # skills must be real categories
    with pytest.raises(ValidationError):
        agent(svc, "Bad", 5, "BOSS")
    a = agent(svc, "A", 5, L1, ["billing"])
    assert a["skills"] == ["billing"] and a["role_label"].startswith("L1")
    with pytest.raises(ValidationError):
        agent(svc, "a", 5)                                          # duplicate name (case-insensitive)
    t = mk(svc, category="billing")["id"]
    with pytest.raises(ValidationError):
        svc.update_agent(a["id"], ROOT, role=SUPERVISOR)            # still holds an active ticket
    svc.transition(t, IN_PROGRESS, a["id"]); svc.transition(t, RESOLVED, a["id"])
    out = svc.update_agent(a["id"], ROOT, role=SUPERVISOR)
    assert out["role"] == SUPERVISOR and out["skills"] == [] and not out["takes_tickets"]
    assert svc.stats()["agents_available"] == 0
    second = agent(svc, "Root2", 1, ADMIN)["id"]
    assert svc.update_agent(ROOT, second, role=L1)["role"] == L1    # fine once another Admin exists


# ============================================================ roles: skill + tier routing
def test_skill_based_routing(svc):
    bill = agent(svc, "Bill", 1, L1, ["billing"])["id"]
    t_login = mk(svc, category="login")["id"]
    assert svc.get_ticket(t_login)["status"] == NEW                 # a billing specialist does not take login
    t_bill = mk(svc, category="billing", priority=4)["id"]
    assert svc.get_ticket(t_bill)["assignee"] == bill               # ... even though login was more urgent/older
    gen = agent(svc, "Gina", 1, L1)["id"]                           # generalist takes anything
    assert svc.get_ticket(t_login)["assignee"] == gen
    t_b2, t_l2 = mk(svc, category="billing")["id"], mk(svc, category="login", priority=1)["id"]
    assert svc.get_ticket(t_b2)["status"] == NEW and svc.get_ticket(t_l2)["status"] == NEW
    svc.transition(t_bill, IN_PROGRESS, bill); svc.transition(t_bill, RESOLVED, bill)
    assert svc.get_ticket(t_b2)["assignee"] == bill                 # specialist pulls only from its own queue
    assert svc.get_ticket(t_l2)["status"] == NEW
    assert svc.stats()["waiting_by_category"] == {"login": 1}
    # specialist is preferred over a free generalist, keeping generalists for uncovered categories
    svc.update_agent(bill, ROOT, capacity=3); svc.update_agent(gen, ROOT, capacity=3)
    assert svc.get_ticket(t_l2)["assignee"] == gen
    assert svc.get_ticket(mk(svc, category="billing")["id"])["assignee"] == bill
    check_all(svc)


def test_tier_preference_with_fallback(svc):
    junior = agent(svc, "Jun", 2, L1)["id"]
    senior = agent(svc, "Sen", 1, L2)["id"]
    assert svc.get_ticket(mk(svc, priority=4)["id"])["assignee"] == junior    # routine work -> L1
    assert svc.get_ticket(mk(svc, priority=1)["id"])["assignee"] == senior    # critical -> L2
    assert svc.get_ticket(mk(svc, priority=1)["id"])["assignee"] == junior    # L2 full -> falls back, never waits
    t = mk(svc, priority=2)["id"]
    assert svc.get_ticket(t)["status"] == NEW                                 # everybody is full
    trail = svc.audit_trail(svc.search(assignee=senior)["tickets"][0]["id"])
    assert '"agent_role":"L2"' in trail[-1]["detail"]


# ============================================================ search
def test_search_matches_bruteforce(svc):
    rnd = random.Random(5)
    cats = ["billing", "login", "outage"]
    subj = ["Payment failed twice", "Cannot reset password", "API timeout error", "Refund pending"]
    for i in range(300):
        mk(svc, rnd.choice(subj), f"user{rnd.randrange(30)}@ex.com", rnd.randrange(1, 5), rnd.choice(cats))
    a, b = agent(svc, "A", 15)["id"], agent(svc, "B", 15, L2, ["billing", "outage"])["id"]
    for _, t in list(svc._by_status.scan((1, -1), (1, 2 ** 62)))[:10]:
        svc.transition(t.id, IN_PROGRESS, ROOT)
    allt = [x.to_dict() for _, x in svc._by_id.items()]

    def brute(**f):
        out = [t for t in allt if
               (not f.get("q") or all(w in (t["subject"] + " " + t["description"] + " " + t["category"]).lower()
                                      for w in f["q"].lower().split())) and
               (not f.get("customer") or t["customer"].startswith(f["customer"])) and
               (not f.get("status") or t["status"] == f["status"]) and
               (not f.get("category") or t["category"] == f["category"]) and
               (not f.get("assignee") or t["assignee"] == f["assignee"]) and
               (not f.get("priority") or t["priority"] == f["priority"])]
        return sorted([t["id"] for t in out], reverse=True)[:f.get("limit", 50)]

    cases = [dict(q="refund"), dict(q="password reset"), dict(q="api error", priority=2),
             dict(customer="user1"), dict(customer="user7@ex.com", status="NEW"), dict(status="IN_PROGRESS"),
             dict(status="NEW", priority=1, limit=7), dict(priority=3), dict(q="nonexistentword"),
             dict(status="ASSIGNED", category="billing"), dict(assignee=b), dict(status="NEW", limit=500),
             dict(q="refund", status="NEW", category="login"), dict(category="outage", limit=5)]
    for c in cases:
        res = svc.search(**c)
        assert [t["id"] for t in res["tickets"]] == brute(**c), (c, res["plan"])
    assert "status" in svc.search(status="NEW")["plan"] and "customer" in svc.search(customer="user1")["plan"]
    mid = allt[150]["created_ms"]
    got = [t["id"] for t in svc.search(created_from=mid, status="NEW", limit=500)["tickets"]]
    assert got == sorted([t["id"] for t in allt if t["created_ms"] >= mid and t["status"] == "NEW"], reverse=True)
    with pytest.raises(ValidationError):
        svc.search(q="a")
    board = svc.board(5)                                            # board columns = newest 5 of each stage
    for s in ("ASSIGNED", "IN_PROGRESS"):
        assert [t["id"] for t in board[s]["tickets"]] == brute(status=s, limit=5)
        assert board[s]["count"] == sum(1 for t in allt if t["status"] == s)
    check_all(svc)


# ============================================================ audit + durability
def test_audit_chain_trigger_and_persistence(svc):
    a = agent(svc, "A", 2, L2)["id"]
    t = mk(svc)["id"]
    svc.transition(t, IN_PROGRESS, a); svc.change_priority(t, 1, a, reason="vip"); svc.transition(t, RESOLVED, a)
    trail = svc.audit_trail(t)
    assert [e["seq"] for e in trail] == [1, 2, 3, 4, 5]
    assert svc.verify_audit(t)["valid"] and svc.verify_all_audits()["valid"]
    assert '"status":["NEW","ASSIGNED"]' in trail[1]["detail"]       # before/after captured
    assert svc.ticket_detail(t)["audit"] == trail
    with pytest.raises(psycopg2.Error):                              # DB-level append-only guard
        with svc.db.tx() as cur:
            cur.execute("UPDATE ticket_audit SET actor='mallory' WHERE ticket_id=%s AND seq=2", (t,))
    with pytest.raises(psycopg2.Error):
        with svc.db.tx() as cur:
            cur.execute("DELETE FROM ticket_audit WHERE ticket_id=%s", (t,))
    mk(svc, category="billing"); mk(svc, priority=1)
    # a restart rebuilds identical state from PostgreSQL
    s2 = TicketRouter(svc.db, btree_order=8)
    s2.load()
    a1, b1 = svc.stats(), s2.stats()
    a1.pop("db_commits"); b1.pop("db_commits")
    assert a1 == b1 and s2.list_agents() == svc.list_agents() and s2.board() == svc.board()
    assert s2.get_ticket(t) == svc.get_ticket(t)
    s2.transition(t, CLOSED, a)                                      # chain continues correctly after reload
    assert s2.verify_audit(t)["valid"] and len(s2.audit_trail(t)) == 6
    check_all(s2)


def test_one_commit_per_request(svc):
    a = agent(svc, "A", 50)["id"]
    c = svc.db.commits
    t = mk(svc)["id"]                                                # ticket + CREATED + AUTO_ASSIGNED
    assert svc.db.commits - c == 1 and svc.get_ticket(t)["status"] == ASSIGNED
    c = svc.db.commits
    r = svc.create_tickets_bulk([{"subject": f"S{i}", "customer": "b@x.com", "category": "bug"} for i in range(40)])
    assert svc.db.commits - c == 1 and r["created"] == 40 and r["auto_assigned"] == 40
    with svc.db.tx() as cur:
        cur.execute("SELECT count(*) FROM tickets"); n_t = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM ticket_audit"); n_a = cur.fetchone()[0]
    assert (n_t, n_a) == (41, 82)
    c = svc.db.commits
    svc.update_agent(a, ROOT, online=False)                          # 41 releases in one transaction
    assert svc.db.commits - c == 1 and svc.stats()["queue_length"] == 41
    with pytest.raises(ValidationError):                             # one bad ticket -> nothing is created
        svc.create_tickets_bulk([{"subject": "ok", "customer": "c@x.com"}, {"subject": "", "customer": "c@x.com"}])
    assert svc.stats()["total_tickets"] == 41
    check_all(svc)


def test_database_failure_leaves_memory_untouched(svc):
    a = agent(svc, "A", 1)["id"]
    t = mk(svc)["id"]                                                # assigned to A
    waiting = mk(svc, "Refund missing", "w@x.com", 1, "billing")["id"]
    before = (svc.stats(), svc.board(), svc.list_agents(), svc.get_ticket(t), svc.get_ticket(waiting))
    orig = svc.db.insert_audits

    def boom(cur, entries):
        raise RuntimeError("db down")
    svc.db.insert_audits = boom
    with pytest.raises(RuntimeError):                                # move + follow-up auto-assign, both undone
        svc.transition(t, IN_PROGRESS, a)
    with pytest.raises(RuntimeError):
        mk(svc, "Brand new zebra issue")                             # creation undone, id not burned
    with pytest.raises(RuntimeError):
        svc.create_tickets_bulk([{"subject": f"Z{i}", "customer": "z@x.com"} for i in range(30)])
    with pytest.raises(RuntimeError):
        svc.update_agent(a, ROOT, online=False)                      # roster change + release undone
    svc.db.insert_audits = orig
    after = (svc.stats(), svc.board(), svc.list_agents(), svc.get_ticket(t), svc.get_ticket(waiting))
    for x, y in zip(before, after):
        if isinstance(x, dict):
            x.pop("db_commits", None); y.pop("db_commits", None)
        assert x == y
    assert svc.search(q="zebra")["count"] == 0 and svc.search(customer="z@x.com")["count"] == 0
    check_all(svc)
    s2 = TicketRouter(svc.db, btree_order=8); s2.load()              # database agrees with memory
    assert s2.board() == svc.board() and s2.verify_all_audits()["valid"]
    nxt = mk(svc)["id"]                                              # works again afterwards
    assert nxt == waiting + 1
    svc.transition(t, IN_PROGRESS, a); svc.transition(t, RESOLVED, a)
    assert svc.get_ticket(waiting)["assignee"] == a


def test_concurrent_creation_is_safe(svc):
    agent(svc, "A", 100)
    errors, ids = [], []

    def worker(n):
        try:
            for i in range(15):
                ids.append(mk(svc, f"Issue {n}-{i}", f"u{n}@x.com", 1 + (i % 4))["id"])
        except Exception as e:           # pragma: no cover
            errors.append(e)
    ths = [threading.Thread(target=worker, args=(n,)) for n in range(12)]
    [t.start() for t in ths]; [t.join() for t in ths]
    assert not errors and len(set(ids)) == 180 == svc.stats()["total_tickets"]
    assert svc.verify_all_audits()["valid"]
    check_all(svc)


# ============================================================ upgrade path
def test_v1_database_is_upgraded_in_place(svc):
    """A database created by the first version (no role / skills / sla_breached columns) keeps working."""
    db = svc.db
    with db.tx() as cur:
        cur.execute("DROP TABLE ticket_audit, tickets, agents")
        cur.execute("""CREATE TABLE agents (id BIGINT PRIMARY KEY, name TEXT NOT NULL UNIQUE, capacity INT NOT NULL,
                           online BOOLEAN NOT NULL DEFAULT TRUE, created_ms BIGINT NOT NULL);
                       CREATE TABLE tickets (id BIGINT PRIMARY KEY, subject TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                           customer TEXT NOT NULL, category TEXT NOT NULL, priority SMALLINT NOT NULL, status TEXT NOT NULL,
                           assignee BIGINT REFERENCES agents(id), created_ms BIGINT NOT NULL, updated_ms BIGINT NOT NULL,
                           sla_due_ms BIGINT NOT NULL, version INT NOT NULL);
                       CREATE TABLE ticket_audit (ticket_id BIGINT NOT NULL REFERENCES tickets(id), seq INT NOT NULL,
                           ts_ms BIGINT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL,
                           prev_hash CHAR(64) NOT NULL, hash CHAR(64) NOT NULL, PRIMARY KEY (ticket_id, seq));
                       INSERT INTO agents VALUES (1, 'Aarav', 2, TRUE, 1);
                       INSERT INTO tickets VALUES (1, 'Old ticket', '', 'o@x.com', 'billing', 3, 'NEW', NULL, 5, 5, 6, 1);""")
    db.init_schema()
    s = TicketRouter(db, btree_order=8); s.load()
    roster = s.list_agents()
    assert [(a["name"], a["role"]) for a in roster] == [("Aarav", L1), ("Admin", ADMIN)]
    s.dispatch()
    assert s.get_ticket(1)["assignee"] == 1 and s.get_ticket(1)["sla_breached"] is False
