"""
seed.py - generate realistic demo data straight into PostgreSQL (bulk COPY, the fastest load path).

    python seed.py --n 100000 --agents 25 --reset

Creates the staff roster (1 Admin, 2 Supervisors, L1/L2 agents with skills), tickets in every
lifecycle stage (consistent with agent capacity and skills) and a valid hash-chained audit
trail for each ticket.
"""
import argparse
import random
import time

import config
from audit import make_entry
from db import Database
from models import ADMIN, ASSIGNED, CLOSED, IN_PROGRESS, L1, L2, NEW, PENDING, RESOLVED, SLA_MS, SUPERVISOR

CATEGORIES = {
    "billing":  ["Double charge on invoice", "Refund not received", "Update payment method", "Invoice shows wrong amount",
                 "Cancel subscription renewal"],
    "login":    ["Cannot log in to account", "Password reset email not arriving", "Two factor code rejected",
                 "Account locked after failed attempts"],
    "outage":   ["Dashboard not loading", "API returns 500 errors", "Service unavailable in region",
                 "Payments failing intermittently"],
    "shipping": ["Order delayed in transit", "Package marked delivered but missing", "Change delivery address",
                 "Damaged item received"],
    "bug":      ["Export to CSV crashes", "Search results incorrect", "Mobile app freezes on startup",
                 "Notification settings not saving"],
    "feature":  ["Request dark mode", "Add bulk edit option", "Integrate with calendar", "Need custom report"],
}
DETAILS = ["Started yesterday and still happening.", "Please escalate, this is blocking our team.",
           "Tried clearing cache and reinstalling.", "Customer has contacted us three times already.",
           "Screenshots attached in the original email.", "Issue affects multiple users in our office."]
AGENT_NAMES = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Isha", "Vihaan", "Ananya", "Arjun", "Sara", "Neel", "Tara",
               "Yash", "Riya", "Dev", "Kiara", "Om", "Nisha", "Parth", "Zoya", "Ishaan", "Pooja", "Raj", "Simran", "Veer",
               "Aditi", "Krish", "Naina", "Samar", "Tanvi"]
# skill sets handed out in turn: 2 of every 5 agents are generalists, so every category is always covered
SKILL_SETS = [(), ("billing",), (), ("login", "bug"), ("outage",), (), ("shipping",), (), ("feature", "bug"),
              ("billing", "shipping")]

PATHS = {
    NEW: [NEW],
    ASSIGNED: [NEW, ASSIGNED],
    IN_PROGRESS: [NEW, ASSIGNED, IN_PROGRESS],
    PENDING: [NEW, ASSIGNED, IN_PROGRESS, PENDING],
    RESOLVED: [NEW, ASSIGNED, IN_PROGRESS, RESOLVED],
    CLOSED: [NEW, ASSIGNED, IN_PROGRESS, RESOLVED, CLOSED],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="number of tickets")
    ap.add_argument("--agents", type=int, default=25, help="number of ticket-handling agents (L1 + L2)")
    ap.add_argument("--capacity", type=int, default=10, help="tickets one agent can work at once")
    ap.add_argument("--reset", action="store_true", help="wipe existing data first")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    db = Database(config.DB)
    db.bootstrap()
    db.init_schema()
    if a.reset:
        db.reset()
    else:
        with db.tx() as cur:
            cur.execute("SELECT (SELECT count(*) FROM tickets), (SELECT count(*) FROM agents)")
            if any(cur.fetchone()):
                raise SystemExit("Database already has data. Re-run with --reset to wipe it first.")

    t0 = time.perf_counter()
    now = int(time.time() * 1000)
    joined = now - 90 * 86_400_000

    # ---- staff roster: id 1 Admin, 2-3 Supervisors, then the agents (every 3rd one is an L2 senior)
    staff = [(1, "Admin", 1, True, joined, ADMIN, ""), (2, "Kavya", 1, True, joined, SUPERVISOR, ""),
             (3, "Rahul", 1, True, joined, SUPERVISOR, "")]
    handlers = {}                                            # id -> (name, skills)
    for i in range(min(a.agents, len(AGENT_NAMES))):
        skills = SKILL_SETS[i % len(SKILL_SETS)]
        aid = len(staff) + 1
        staff.append((aid, AGENT_NAMES[i], a.capacity, True, joined, L2 if i % 3 == 2 else L1, ",".join(sorted(skills))))
        handlers[aid] = (AGENT_NAMES[i], frozenset(skills))
    load = {aid: 0 for aid in handlers}
    can_take = {c: [aid for aid, (_, sk) in handlers.items() if not sk or c in sk] for c in CATEGORIES}
    handler_ids = list(handlers)
    slots = len(handlers) * a.capacity

    # ---- decide each ticket's final status, respecting agent capacity
    n_closed, n_resolved = int(a.n * 0.60), int(a.n * 0.15)
    n_open = a.n - n_closed - n_resolved
    n_working = min(slots, int(n_open * 0.5)) if n_open else 0
    statuses = [CLOSED] * n_closed + [RESOLVED] * n_resolved
    statuses += [rnd.choice((ASSIGNED, IN_PROGRESS, IN_PROGRESS, PENDING)) for _ in range(n_working)]
    statuses += [NEW] * (a.n - len(statuses))

    # ---- draft tickets (created time depends on status; open tickets are recent so SLAs are realistic)
    n_customers = max(50, a.n // 15)
    drafts = []
    for st in statuses:
        pri = rnd.choices((1, 2, 3, 4), weights=(5, 15, 50, 30))[0]
        if st in (CLOSED, RESOLVED):
            created = now - rnd.randrange(3 * 3_600_000, 90 * 86_400_000)
        else:      # open: ~5% already slightly past SLA so escalation is visible straight away
            created = now - int(rnd.uniform(0, 1.05) * SLA_MS[pri])
        drafts.append((created, st, pri))
    drafts.sort(key=lambda d: d[0])                  # ids increase with creation time

    tickets, audit_rows, counts = [], [], {}
    cat_names = list(CATEGORIES)
    for idx, (created, st, pri) in enumerate(drafts):
        tid = idx + 1
        cat = rnd.choice(cat_names)
        subject = rnd.choice(CATEGORIES[cat])
        desc = " ".join(rnd.sample(DETAILS, 2))
        customer = f"user{rnd.randrange(n_customers)}@example.com"
        assignee = None
        if st in (ASSIGNED, IN_PROGRESS, PENDING):   # only agents with a free slot AND the right skill
            free = [aid for aid in can_take[cat] if load[aid] < a.capacity]
            if free:
                assignee = rnd.choice(free)
                load[assignee] += 1
            else:
                st = NEW                             # nobody suitable is free -> it waits in the queue
        elif st in (RESOLVED, CLOSED):
            assignee = rnd.choice(can_take[cat] or handler_ids)
        path = PATHS[st]
        # audit chain: timestamps strictly increasing, never in the future
        ts, prev = created, None
        for k, s in enumerate(path):
            if k == 0:
                e = make_entry(prev, tid, ts, "customer", "CREATED",
                               {"priority": pri, "category": cat, "customer": customer, "version": 1})
            else:
                ts = min(now, ts + rnd.randrange(60_000, 3 * 3_600_000))
                if s == ASSIGNED:
                    e = make_entry(prev, tid, ts, "system", "AUTO_ASSIGNED",
                                   {"agent": handlers[assignee][0], "agent_id": assignee,
                                    "changes": {"assignee": [None, assignee], "status": [NEW, ASSIGNED]}, "version": k + 1})
                else:
                    e = make_entry(prev, tid, ts, handlers[assignee][0], "STATUS_CHANGED",
                                   {"changes": {"status": [path[k - 1], s]}, "version": k + 1})
            audit_rows.append(tuple(e))
            prev = (e.seq, e.hash)
        counts[st] = counts.get(st, 0) + 1
        tickets.append((tid, subject, desc, customer, cat, pri, st, assignee, created, ts,
                        created + SLA_MS[pri], len(path), False))
    t_gen = time.perf_counter() - t0

    # ---- write (one transaction, COPY)
    with db.tx() as cur:
        db.copy_rows(cur, "agents", staff)
        db.copy_rows(cur, "tickets", tickets)
        db.copy_rows(cur, "ticket_audit", audit_rows)
    print(f"Seeded {len(tickets):,} tickets, {len(staff)} staff, {len(audit_rows):,} audit entries "
          f"in {time.perf_counter() - t0:.1f}s (generate {t_gen:.1f}s + database {time.perf_counter() - t0 - t_gen:.1f}s)")
    print("  by status:", counts)
    print("  staff: id 1 = Admin, ids 2-3 = Supervisors, the rest = L1 / L2 agents")
    db.close()


if __name__ == "__main__":
    main()
