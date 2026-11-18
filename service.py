"""
service.py - TicketRouter: the engine that ties every data structure together.

IN-MEMORY STRUCTURES (rebuilt from PostgreSQL at start-up)
  _by_id        B+ tree   ticket_id               -> Ticket   primary index
  _by_customer  B+ tree   (customer, ticket_id)   -> Ticket   exact + PREFIX customer search (range scan)
  _by_time      B+ tree   (created_ms, ticket_id) -> Ticket   time-range queries, newest first (reverse scan)
  _by_status    B+ tree   (status, ticket_id)     -> Ticket   lifecycle stages: "latest N in a stage" = O(log n + N)
  _inverted     dict[token] -> set(ids)                       inverted index for free-text search (AND)
  _queue        IndexedMinHeap key=(priority, sla_due, id)    all waiting tickets, most urgent first
  _cat_queue    dict[category] -> IndexedMinHeap              same key, one heap per category (skill routing)
  _sla          IndexedMinHeap key=(sla_due, id)              earliest deadline first -> breach detection
  _pools        dict[(role, skill)] -> IndexedMinHeap key=(load, agent_id)
                                                              least-loaded AVAILABLE agent per tier and skill
  _agent_tickets dict[agent_id] -> set(ids)                   agent workload, O(1) load = len(set)
  _audit_tail   dict[ticket_id] -> (last_seq, last_hash)      O(1) append to the hash chain

ROUTING (event driven, never scans a list)
  ticket starts waiting -> _pick_agent : peek at most 4 pool heaps (preferred tier first, specialist first)
  agent gets a free slot -> _pick_ticket: peek the queue heap(s) the agent can serve, take the most urgent

WRITES: one "unit of work" per API request
  Changes are applied to memory as they happen and collected. When the request ends, ALL ticket rows
  and audit rows are written to PostgreSQL in ONE transaction (one commit). If that fails, every memory
  change is undone in reverse order - memory and database can never disagree, and no other request
  sees the intermediate state because the lock is held.

Concurrency: one re-entrant lock guards all in-memory state (run uvicorn with a single worker process).
"""
import heapq
import logging
import re
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Set, Tuple

from audit import AuditEntry, make_entry, verify_chain
from bplus_tree import BPlusTree
from db import Database
from indexed_heap import IndexedMinHeap
from models import (ADMIN, ANY_SKILL, ASSIGNED, CATEGORIES, CLOSED, HANDLER_ROLES, L1, NEW, P_ASSIGN,
                    P_AVAIL_ANY, P_AVAIL_SELF, P_PRIORITY_ANY, P_PRIORITY_OWN, P_ROSTER, P_WORK_ANY,
                    P_WORK_OWN, PERMISSION_TEXT, PERMISSIONS, PRIORITY_NAMES, RESOLVED, ROLE_LABELS,
                    ROLES, SLA_MS, SLA_RUNNING, STATUS_INDEX, STATUSES, TIER_ORDER, TRANSITIONS, WORKING,
                    Agent, CapacityError, ForbiddenError, InvalidTransitionError, NotFoundError, Ticket,
                    ValidationError)

log = logging.getLogger("router")

_WORD = re.compile(r"[a-z0-9]{2,}")
_STOP = frozenset({"the", "and", "for", "with", "not", "are", "was", "has", "have", "can", "you",
                   "this", "that", "from", "but", "all", "its", "our", "your", "any"})
_BIG = 2 ** 63
_MAX_CHAR = "\U0010ffff"


def now_ms() -> int:
    return int(time.time() * 1000)


def tokenize(text: str) -> Set[str]:
    return set(_WORD.findall(text.lower())) - _STOP


class _Unit:
    """Everything one request changed: rows to write at commit + how to undo the memory changes."""
    __slots__ = ("tickets", "audits", "agents", "undo")

    def __init__(self) -> None:
        self.tickets: Dict[int, Ticket] = {}
        self.audits: List[AuditEntry] = []
        self.agents: Dict[int, Agent] = {}
        self.undo: list = []


class TicketRouter:
    def __init__(self, db: Database, btree_order: int = 64) -> None:
        self.db = db
        self._lock = threading.RLock()
        self._by_id = BPlusTree(btree_order)
        self._by_customer = BPlusTree(btree_order)
        self._by_time = BPlusTree(btree_order)
        self._by_status = BPlusTree(btree_order)
        self._status_count: Dict[str, int] = {s: 0 for s in STATUSES}
        self._inverted: Dict[str, Set[int]] = {}
        self._queue = IndexedMinHeap()
        self._cat_queue: Dict[str, IndexedMinHeap] = {}
        self._sla = IndexedMinHeap()
        self._pools: Dict[Tuple[str, str], IndexedMinHeap] = {}
        self._pool_keys: Dict[int, tuple] = {}          # agent_id -> pool keys it currently sits in
        self._available: Set[int] = set()               # handler agents that are online with a free slot
        self._agents: Dict[int, Agent] = {}
        self._agent_tickets: Dict[int, Set[int]] = defaultdict(set)
        self._audit_tail: Dict[int, tuple] = {}
        self._released_by: Dict[int, int] = {}          # waiting ticket -> agent who handed it back
        self._next_ticket_id = 1
        self._next_agent_id = 1
        self._uow: Optional[_Unit] = None
        self._dirty_tickets: Set[int] = set()           # tickets that just started waiting
        self._dirty_agents: Set[int] = set()            # agents that may have a new free slot
        self._assigned_log: List[Tuple[int, int]] = []
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # ================================================================ start-up
    def load(self) -> None:
        """Rebuild every in-memory structure from PostgreSQL. Uses O(n) bulk builders."""
        t0 = time.perf_counter()
        with self._lock:
            for id_, name, cap, online, created, role, skills in self.db.load_agents():
                role = role if role in ROLES else L1
                self._agents[id_] = Agent(id_, name, cap, online, created, role,
                                          frozenset(s for s in skills.split(",") if s))
            tickets: List[Ticket] = [Ticket(*row) for row in self.db.iter_tickets()]   # sorted by id
            self._by_id.bulk_load((t.id, t) for t in tickets)
            self._by_customer.bulk_load(((t.customer, t.id), t)
                                        for t in sorted(tickets, key=lambda x: (x.customer, x.id)))
            self._by_time.bulk_load(((t.created_ms, t.id), t)
                                    for t in sorted(tickets, key=lambda x: (x.created_ms, x.id)))
            self._by_status.bulk_load(((STATUS_INDEX[t.status], t.id), t)
                                      for t in sorted(tickets, key=lambda x: STATUS_INDEX[x.status]))  # stable
            queue_pairs, sla_pairs = [], []
            cat_pairs: Dict[str, list] = defaultdict(list)
            for t in tickets:
                self._status_count[t.status] += 1
                self._index_text(t)
                if t.status == NEW:
                    pair = (t.id, (t.priority, t.sla_due_ms, t.id))
                    queue_pairs.append(pair)
                    cat_pairs[t.category].append(pair)
                if t.status in SLA_RUNNING and not t.sla_breached:
                    sla_pairs.append((t.id, (t.sla_due_ms, t.id)))
                if t.status in WORKING and t.assignee is not None:
                    self._agent_tickets[t.assignee].add(t.id)
            self._queue.build(queue_pairs)
            for cat, pairs in cat_pairs.items():
                self._cat_queue[cat] = IndexedMinHeap()
                self._cat_queue[cat].build(pairs)
            self._sla.build(sla_pairs)
            for a in self._agents.values():
                self._refresh_agent(a.id)
            for tid, seq, h in self.db.load_audit_tails():
                self._audit_tail[tid] = (seq, h)
            if tickets:
                self._next_ticket_id = tickets[-1].id + 1
            if self._agents:
                self._next_agent_id = max(self._agents) + 1
            self._ensure_admin()
        log.info("loaded %d tickets, %d agents in %.2fs", len(self._by_id), len(self._agents),
                 time.perf_counter() - t0)

    def _ensure_admin(self) -> None:
        """A roster with no Admin could never be managed, so create one on first start."""
        if any(a.role == ADMIN for a in self._agents.values()):
            return
        taken = {a.name.lower() for a in self._agents.values()}
        name = next(n for n in ("Admin", "System Admin", f"Admin {self._next_agent_id}") if n.lower() not in taken)
        a = Agent(self._next_agent_id, name, 1, True, now_ms(), ADMIN, frozenset())
        with self.db.tx() as cur:
            self.db.upsert_agents(cur, [a])
        self._agents[a.id] = a
        self._next_agent_id += 1
        log.info("created default administrator '%s' (id %d)", a.name, a.id)

    # ================================================================ background worker
    def start_background(self, interval_s: float = 5.0) -> None:
        def run():
            while not self._stop.wait(interval_s):
                try:
                    self.escalate_breached()
                    self.dispatch()
                except Exception:           # keep the worker alive no matter what
                    log.exception("background worker error")
        self._worker = threading.Thread(target=run, name="sla-dispatch-worker", daemon=True)
        self._worker.start()

    def stop_background(self) -> None:
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=5)

    # ================================================================ unit of work
    @contextmanager
    def _unit(self):
        """Call with the lock held. Nested use joins the outer unit (still one commit)."""
        if self._uow is not None:
            yield
            return
        uow = self._uow = _Unit()
        try:
            yield
            self._settle()                                   # route whatever this request freed or queued
            if uow.tickets or uow.audits or uow.agents:
                with self.db.tx() as cur:                    # ONE transaction, ONE commit
                    if uow.agents:
                        self.db.upsert_agents(cur, uow.agents.values())
                    if uow.tickets:
                        self.db.upsert_tickets(cur, uow.tickets.values())
                    if uow.audits:
                        self.db.insert_audits(cur, uow.audits)
        except BaseException:
            for undo in reversed(uow.undo):                  # database said no -> memory goes back
                undo()
            self._dirty_tickets.clear()
            self._dirty_agents.clear()
            raise
        finally:
            self._uow = None

    # ================================================================ internal helpers
    def _get(self, ticket_id: int) -> Ticket:
        t = self._by_id.get(ticket_id)
        if t is None:
            raise NotFoundError(f"ticket {ticket_id} not found")
        return t

    def _get_agent(self, agent_id: int) -> Agent:
        a = self._agents.get(agent_id)
        if a is None:
            raise NotFoundError(f"agent {agent_id} not found")
        return a

    def _index_text(self, t: Ticket) -> None:
        for tok in tokenize(t.subject + " " + t.description + " " + t.category):
            self._inverted.setdefault(tok, set()).add(t.id)

    def _unindex_text(self, t: Ticket) -> None:
        for tok in tokenize(t.subject + " " + t.description + " " + t.category):
            ids = self._inverted.get(tok)
            if ids is not None:
                ids.discard(t.id)
                if not ids:
                    del self._inverted[tok]

    def _cat_heap(self, category: str) -> IndexedMinHeap:
        h = self._cat_queue.get(category)
        if h is None:
            h = self._cat_queue[category] = IndexedMinHeap()
        return h

    # ---------------------------------------------------------------- permissions
    def _actor(self, actor_id: Optional[int]) -> Agent:
        if actor_id is None:
            raise ForbiddenError("this action needs a signed-in staff member (X-Actor-Id header)")
        a = self._agents.get(actor_id)
        if a is None:
            raise ForbiddenError(f"unknown staff id {actor_id}")
        return a

    @staticmethod
    def _require(actor: Agent, perm: str) -> None:
        if perm not in PERMISSIONS[actor.role]:
            raise ForbiddenError(f"{actor.name} ({ROLE_LABELS[actor.role]}) is not allowed to {PERMISSION_TEXT[perm]}")

    @staticmethod
    def _require_ticket(actor: Agent, t: Ticket, own_perm: str, any_perm: str) -> None:
        perms = PERMISSIONS[actor.role]
        if any_perm in perms or (own_perm in perms and t.assignee == actor.id):
            return
        why = "it is not assigned to them" if own_perm in perms else "their role does not allow it"
        raise ForbiddenError(f"{actor.name} ({ROLE_LABELS[actor.role]}) cannot change ticket #{t.id}: {why}")

    def check(self, actor_id: Optional[int], perm: str) -> None:
        with self._lock:
            self._require(self._actor(actor_id), perm)

    # ---------------------------------------------------------------- agent pools
    def _refresh_agent(self, agent_id: int) -> None:
        """Keep the pools exact: an agent sits in its pool heaps iff it can take a ticket right now."""
        a = self._agents[agent_id]
        load = len(self._agent_tickets[agent_id])
        avail = a.role in HANDLER_ROLES and a.online and load < a.capacity
        want = tuple((a.role, s) for s in (a.skills or (ANY_SKILL,))) if avail else ()
        for key in self._pool_keys.get(agent_id, ()):
            if key not in want:
                self._pools[key].discard(agent_id)
        for key in want:
            pool = self._pools.get(key)
            if pool is None:
                pool = self._pools[key] = IndexedMinHeap()
            pool.push(agent_id, (load, agent_id))            # insert OR re-key
        self._pool_keys[agent_id] = want
        if avail:
            self._available.add(agent_id)
        else:
            self._available.discard(agent_id)

    # ---------------------------------------------------------------- routing
    def _pick_agent(self, t: Ticket) -> Optional[int]:
        """Best free agent for a waiting ticket: preferred tier first, specialist before generalist,
        least loaded inside the pool. At most 4 heap peeks."""
        blocked = self._released_by.get(t.id)
        for role in TIER_ORDER[t.priority]:
            for skill in (t.category, ANY_SKILL):
                pool = self._pools.get((role, skill))
                if not pool:
                    continue
                if blocked is None:
                    return pool.peek()[0]
                for aid, _ in pool.peek_n(2):
                    if aid != blocked:
                        return aid
        return None

    def _pick_ticket(self, a: Agent) -> Optional[int]:
        """Most urgent waiting ticket this agent may serve. A generalist looks at the global heap;
        a specialist compares the tops of its skill heaps."""
        heaps: Iterable[IndexedMinHeap] = ((self._queue,) if not a.skills else
                                           (self._cat_queue[s] for s in a.skills if s in self._cat_queue))
        depth = 1 + min(len(self._released_by), 50)          # skip tickets this agent handed back
        best = best_key = None
        for h in heaps:
            if not h:
                continue
            for tid, key in ((h.peek(),) if depth == 1 else h.peek_n(depth)):
                if self._released_by.get(tid) == a.id:
                    continue
                if best is None or key < best_key:
                    best, best_key = tid, key
                break
        return best

    def _auto_assign(self, t: Ticket, agent_id: int) -> None:
        a = self._agents[agent_id]
        self._change(t, None, "AUTO_ASSIGNED",
                     {"agent_id": a.id, "agent": a.name, "agent_role": a.role,
                      "agent_load_before": len(self._agent_tickets[a.id])},
                     status=ASSIGNED, assignee=a.id)
        self._assigned_log.append((t.id, a.id))

    def _settle(self) -> None:
        """Restore the invariant 'no ticket waits while a suitable agent is free' after any change."""
        tickets, agents = self._dirty_tickets, self._dirty_agents
        while tickets or agents:
            if tickets:
                t = self._by_id.get(tickets.pop())
                if t is not None and t.status == NEW:
                    aid = self._pick_agent(t)
                    if aid is not None:
                        self._auto_assign(t, aid)
            else:
                aid = agents.pop()
                while aid in self._available:
                    tid = self._pick_ticket(self._agents[aid])
                    if tid is None:
                        break
                    self._auto_assign(self._by_id.get(tid), aid)

    # ---------------------------------------------------------------- the single write path
    def _apply(self, t: Ticket, nt: Ticket) -> None:
        """Copy new state into the live ticket and maintain every index. All O(log n) or O(1).
        It is its own inverse: _apply(t, old_snapshot) undoes a change."""
        old_status, old_asg = t.status, t.assignee
        t.status, t.assignee, t.priority = nt.status, nt.assignee, nt.priority
        t.sla_due_ms, t.updated_ms, t.version, t.sla_breached = nt.sla_due_ms, nt.updated_ms, nt.version, nt.sla_breached

        if old_status != t.status:
            self._by_status.delete((STATUS_INDEX[old_status], t.id))
            self._by_status.insert((STATUS_INDEX[t.status], t.id), t)
            self._status_count[old_status] -= 1
            self._status_count[t.status] += 1
        if t.status == NEW:
            key = (t.priority, t.sla_due_ms, t.id)
            self._queue.push(t.id, key)                      # insert OR re-key
            self._cat_heap(t.category).push(t.id, key)
            if old_status != NEW:
                self._dirty_tickets.add(t.id)
        else:
            self._released_by.pop(t.id, None)
            if old_status == NEW:
                self._queue.discard(t.id)
                self._cat_heap(t.category).discard(t.id)
        if t.status in SLA_RUNNING and not t.sla_breached:
            self._sla.push(t.id, (t.sla_due_ms, t.id))
        else:
            self._sla.discard(t.id)
        old_agent = old_asg if old_status in WORKING else None
        new_agent = t.assignee if t.status in WORKING else None
        if old_agent != new_agent:
            if old_agent is not None:
                self._agent_tickets[old_agent].discard(t.id)
                self._refresh_agent(old_agent)
                self._dirty_agents.add(old_agent)            # a slot just became free
            if new_agent is not None:
                self._agent_tickets[new_agent].add(t.id)
                self._refresh_agent(new_agent)

    def _change(self, t: Ticket, actor: Optional[Agent], action: str, extra: dict, **changes) -> AuditEntry:
        """Every ticket change goes through here: memory + audit entry now, database at unit commit."""
        with self._unit():
            old, old_tail = t.copy(), self._audit_tail.get(t.id)
            nt = t.copy()
            diff = {}
            for field, new_val in changes.items():
                old_val = getattr(t, field)
                if old_val != new_val:
                    diff[field] = [old_val, new_val]
                setattr(nt, field, new_val)
            nt.version = t.version + 1
            nt.updated_ms = now_ms()
            detail = dict(extra)
            detail["changes"] = diff
            detail["version"] = nt.version
            if actor is not None:
                detail["actor_role"] = actor.role
            entry = make_entry(old_tail, t.id, nt.updated_ms, actor.name if actor else "system", action, detail)
            self._apply(t, nt)
            self._audit_tail[t.id] = (entry.seq, entry.hash)
            uow = self._uow
            uow.tickets[t.id] = t
            uow.audits.append(entry)

            def undo() -> None:
                self._apply(t, old)
                if old_tail is None:
                    self._audit_tail.pop(t.id, None)
                else:
                    self._audit_tail[t.id] = old_tail
            uow.undo.append(undo)
            return entry

    def _insert_new(self, t: Ticket, entry: AuditEntry) -> None:
        self._by_id.insert(t.id, t)
        self._by_customer.insert((t.customer, t.id), t)
        self._by_time.insert((t.created_ms, t.id), t)
        self._by_status.insert((STATUS_INDEX[NEW], t.id), t)
        self._status_count[NEW] += 1
        self._index_text(t)
        key = (t.priority, t.sla_due_ms, t.id)
        self._queue.push(t.id, key)
        self._cat_heap(t.category).push(t.id, key)
        self._sla.push(t.id, (t.sla_due_ms, t.id))
        self._audit_tail[t.id] = (entry.seq, entry.hash)
        self._next_ticket_id = t.id + 1
        self._dirty_tickets.add(t.id)
        uow = self._uow
        uow.tickets[t.id] = t
        uow.audits.append(entry)

        def undo() -> None:                                  # ticket is back in NEW by the time this runs
            self._by_id.delete(t.id)
            self._by_customer.delete((t.customer, t.id))
            self._by_time.delete((t.created_ms, t.id))
            self._by_status.delete((STATUS_INDEX[t.status], t.id))
            self._status_count[t.status] -= 1
            self._unindex_text(t)
            self._queue.discard(t.id)
            self._cat_heap(t.category).discard(t.id)
            self._sla.discard(t.id)
            self._audit_tail.pop(t.id, None)
            self._next_ticket_id = t.id
        uow.undo.append(undo)

    # ================================================================ agents / roster
    @staticmethod
    def _clean_skills(skills, role: str) -> frozenset:
        if role not in HANDLER_ROLES or not skills:
            return frozenset()
        out = frozenset(str(s).strip().lower() for s in skills if str(s).strip())
        bad = sorted(out - set(CATEGORIES))
        if bad:
            raise ValidationError(f"unknown skill(s) {', '.join(bad)}. Allowed: {', '.join(CATEGORIES)}")
        return out

    def create_agent(self, name: str, capacity: int = 5, role: str = L1, skills=None,
                     actor_id: Optional[int] = None) -> dict:
        name = (name or "").strip()
        role = (role or L1).upper()
        if not name:
            raise ValidationError("agent name is required")
        if not 1 <= capacity <= 100:
            raise ValidationError("capacity must be 1..100")
        if role not in ROLES:
            raise ValidationError(f"role must be one of {', '.join(ROLES)}")
        with self._lock:
            self._require(self._actor(actor_id), P_ROSTER)
            if any(a.name.lower() == name.lower() for a in self._agents.values()):
                raise ValidationError(f"agent '{name}' already exists")
            a = Agent(self._next_agent_id, name, capacity, True, now_ms(), role, self._clean_skills(skills, role))
            with self._unit():
                self._agents[a.id] = a
                self._next_agent_id = a.id + 1
                self._refresh_agent(a.id)
                self._uow.agents[a.id] = a
                self._dirty_agents.add(a.id)                 # new capacity is used straight away

                def undo() -> None:
                    for key in self._pool_keys.pop(a.id, ()):
                        self._pools[key].discard(a.id)
                    self._available.discard(a.id)
                    self._agents.pop(a.id, None)
                    self._agent_tickets.pop(a.id, None)
                    self._next_agent_id = a.id
                self._uow.undo.append(undo)
            return a.to_dict(len(self._agent_tickets[a.id]))

    def update_agent(self, agent_id: int, actor_id: Optional[int] = None, online: Optional[bool] = None,
                     capacity: Optional[int] = None, role: Optional[str] = None, skills=None) -> dict:
        with self._lock:
            actor = self._actor(actor_id)
            a = self._get_agent(agent_id)
            perms = PERMISSIONS[actor.role]
            new_online = a.online if online is None else bool(online)
            new_cap = a.capacity if capacity is None else capacity
            new_role = a.role if role is None else role.upper()
            if new_role not in ROLES:
                raise ValidationError(f"role must be one of {', '.join(ROLES)}")
            new_skills = a.skills if skills is None else self._clean_skills(skills, new_role)
            if new_role not in HANDLER_ROLES:
                new_skills = frozenset()
            if new_online != a.online and not (P_AVAIL_ANY in perms or (P_AVAIL_SELF in perms and actor.id == a.id)):
                self._require(actor, P_AVAIL_ANY)
            if (new_cap, new_role, new_skills) != (a.capacity, a.role, a.skills):
                self._require(actor, P_ROSTER)
            if not 1 <= new_cap <= 100:
                raise ValidationError("capacity must be 1..100")
            if new_role != a.role:
                if a.role == ADMIN and sum(1 for x in self._agents.values() if x.role == ADMIN) == 1:
                    raise ValidationError("cannot change the role of the last Admin")
                if new_role not in HANDLER_ROLES and self._agent_tickets[a.id]:
                    raise ValidationError(f"{a.name} still has {len(self._agent_tickets[a.id])} active ticket(s): "
                                          f"re-assign them before making them a {ROLE_LABELS[new_role]}")
            with self._unit():
                old = (a.online, a.capacity, a.role, a.skills)
                went_offline = a.online and not new_online
                a.online, a.capacity, a.role, a.skills = new_online, new_cap, new_role, new_skills
                self._refresh_agent(a.id)
                self._uow.agents[a.id] = a

                def undo() -> None:
                    a.online, a.capacity, a.role, a.skills = old
                    self._refresh_agent(a.id)
                self._uow.undo.append(undo)
                if went_offline:      # tickets not yet started go back to the queue
                    for tid in sorted(self._agent_tickets[a.id]):
                        t = self._by_id.get(tid)
                        if t.status == ASSIGNED:
                            self._released_by[t.id] = a.id
                            self._change(t, None, "RELEASED_AGENT_OFFLINE", {"agent_id": a.id, "agent": a.name},
                                         status=NEW, assignee=None)
                self._dirty_agents.add(a.id)
            return a.to_dict(len(self._agent_tickets[a.id]))

    def list_agents(self) -> List[dict]:
        with self._lock:
            return [a.to_dict(len(self._agent_tickets[a.id]))
                    for a in sorted(self._agents.values(), key=lambda x: x.id)]

    # ================================================================ ticket lifecycle
    def _create(self, subject: str, description: str, customer: str, category: str, priority: int,
                actor: str) -> Ticket:
        subject, customer = (subject or "").strip(), (customer or "").strip().lower()
        category = (category or "general").strip().lower()
        if not subject or len(subject) > 200:
            raise ValidationError("subject is required (max 200 chars)")
        if not customer:
            raise ValidationError("customer is required")
        if category not in CATEGORIES:
            raise ValidationError(f"category must be one of {', '.join(CATEGORIES)}")
        if priority not in PRIORITY_NAMES:
            raise ValidationError("priority must be 1 (critical) .. 4 (low)")
        now = now_ms()
        t = Ticket(self._next_ticket_id, subject, (description or "").strip(), customer, category,
                   priority, NEW, None, now, now, now + SLA_MS[priority], 1, False)
        entry = make_entry(None, t.id, now, actor, "CREATED",
                           {"priority": priority, "category": category, "customer": customer, "version": 1})
        self._insert_new(t, entry)
        return t

    def create_ticket(self, subject: str, description: str, customer: str, category: str = "general",
                      priority: int = 3, actor: str = "customer") -> dict:
        with self._lock:
            with self._unit():           # ticket row + CREATED + (if an agent is free) AUTO_ASSIGNED = 1 commit
                t = self._create(subject, description, customer, category, priority, actor)
            return t.to_dict()

    def create_tickets_bulk(self, items: List[dict], actor: str = "customer") -> dict:
        """Create many tickets in ONE transaction (all or nothing)."""
        if not 1 <= len(items) <= 1000:
            raise ValidationError("send between 1 and 1000 tickets")
        with self._lock:
            with self._unit():
                made = [self._create(i.get("subject", ""), i.get("description", ""), i.get("customer", ""),
                                     i.get("category", "general"), i.get("priority", 3), actor) for i in items]
            return {"created": len(made), "first_id": made[0].id, "last_id": made[-1].id,
                    "auto_assigned": sum(1 for t in made if t.status == ASSIGNED)}

    def get_ticket(self, ticket_id: int) -> dict:
        with self._lock:
            return self._get(ticket_id).to_dict()

    def transition(self, ticket_id: int, to_status: str, actor_id: Optional[int] = None, note: str = "") -> dict:
        to_status = (to_status or "").upper()
        if to_status not in STATUSES:
            raise ValidationError(f"unknown status '{to_status}'")
        if to_status == ASSIGNED:
            raise ValidationError("use the assign endpoint to assign a ticket to an agent")
        with self._lock:
            actor = self._actor(actor_id)
            t = self._get(ticket_id)
            self._require_ticket(actor, t, P_WORK_OWN, P_WORK_ANY)
            if to_status not in TRANSITIONS[t.status]:
                allowed = sorted(TRANSITIONS[t.status] - {ASSIGNED}) or ["(none - terminal)"]
                raise InvalidTransitionError(f"{t.status} -> {to_status} is not allowed. Allowed: {', '.join(allowed)}")
            changes: dict = {"status": to_status}
            if to_status == NEW:
                changes["assignee"] = None
            if t.status not in SLA_RUNNING and to_status in SLA_RUNNING:     # clock restarts
                changes["sla_due_ms"] = now_ms() + SLA_MS[t.priority]
                changes["sla_breached"] = False
            action = ("REOPENED" if t.status == RESOLVED and to_status == NEW else
                      "RELEASED" if t.status == ASSIGNED and to_status == NEW else "STATUS_CHANGED")
            with self._unit():           # the move + any ticket the freed slot picks up = 1 commit
                if action == "RELEASED":
                    self._released_by[t.id] = t.assignee      # do not hand it straight back to the same agent
                self._change(t, actor, action, {"note": note} if note else {}, **changes)
            return t.to_dict()

    def assign(self, ticket_id: int, agent_id: int, actor_id: Optional[int] = None) -> dict:
        """Manual assignment (waiting tickets) or re-assignment (already being worked)."""
        with self._lock:
            actor = self._actor(actor_id)
            self._require(actor, P_ASSIGN)
            t = self._get(ticket_id)
            a = self._get_agent(agent_id)
            if t.status != NEW and t.status not in WORKING:
                raise InvalidTransitionError(f"cannot assign a {t.status} ticket")
            if a.role not in HANDLER_ROLES:
                raise ValidationError(f"{a.name} is a {ROLE_LABELS[a.role]} and does not take tickets")
            if t.assignee == a.id and t.status in WORKING:
                raise ValidationError("ticket is already assigned to this agent")
            if not a.online:
                raise CapacityError(f"agent {a.name} is offline")
            if len(self._agent_tickets[a.id]) >= a.capacity:
                raise CapacityError(f"agent {a.name} is at capacity ({a.capacity})")
            extra = {"agent": a.name, "agent_role": a.role}
            if a.skills and t.category not in a.skills:
                extra["skill_override"] = True                # supervisor decision, visible in the audit trail
            if t.status == NEW:
                self._change(t, actor, "MANUAL_ASSIGNED", extra, status=ASSIGNED, assignee=a.id)
            else:
                self._change(t, actor, "REASSIGNED", extra, assignee=a.id)
            return t.to_dict()

    def change_priority(self, ticket_id: int, priority: int, actor_id: Optional[int] = None, reason: str = "") -> dict:
        if priority not in PRIORITY_NAMES:
            raise ValidationError("priority must be 1 (critical) .. 4 (low)")
        with self._lock:
            actor = self._actor(actor_id)
            t = self._get(ticket_id)
            self._require_ticket(actor, t, P_PRIORITY_OWN, P_PRIORITY_ANY)
            if t.status in (RESOLVED, CLOSED):
                raise InvalidTransitionError(f"cannot change priority of a {t.status} ticket")
            if t.priority == priority:
                raise ValidationError("ticket already has this priority")
            changes = {"priority": priority}
            if t.status in SLA_RUNNING:
                changes["sla_due_ms"] = now_ms() + SLA_MS[priority]
                changes["sla_breached"] = False
            self._change(t, actor, "PRIORITY_CHANGED", {"reason": reason} if reason else {}, **changes)
            return t.to_dict()

    # ================================================================ dispatching & SLA
    def dispatch(self) -> List[dict]:
        """Full sweep (safety net, also run by the background worker): let every free agent pull work.
        Normal operation never needs it because routing is event driven."""
        with self._lock:
            self._assigned_log = []
            with self._unit():
                self._dirty_agents.update(self._available)
            return [{"ticket_id": tid, "agent_id": aid} for tid, aid in self._assigned_log]

    def escalate_breached(self, at_ms: Optional[int] = None, max_items: int = 50000, batch: int = 500) -> List[dict]:
        """Handle every ticket whose SLA deadline has passed (heap top = earliest deadline).
        Cost O(b log n) for b overdue tickets - healthy tickets are never touched.
        Written in batches: one commit per `batch` tickets, and the lock is released between batches
        so API requests are never stuck behind a large sweep."""
        at_ms = now_ms() if at_ms is None else at_ms
        out: List[dict] = []
        while len(out) < max_items:
            done = 0
            with self._lock:
                with self._unit():
                    while self._sla and done < batch:
                        tid, (due, _) = self._sla.peek()
                        if due > at_ms:
                            break
                        t = self._by_id.get(tid)
                        if t.priority > 1:                   # escalate one level, new deadline
                            new_p = t.priority - 1
                            self._change(t, None, "SLA_ESCALATED",
                                         {"from_priority": t.priority, "to_priority": new_p, "overdue_ms": at_ms - due},
                                         priority=new_p, sla_due_ms=at_ms + SLA_MS[new_p])
                            out.append({"ticket_id": tid, "action": "ESCALATED", "new_priority": new_p})
                        else:                                # already P1: record the breach ONCE, stop tracking
                            self._change(t, None, "SLA_BREACHED", {"priority": 1, "overdue_ms": at_ms - due},
                                         sla_breached=True)
                            out.append({"ticket_id": tid, "action": "BREACHED_P1", "new_priority": 1})
                        done += 1
            if done < batch:
                break
            time.sleep(0.002)        # Python locks are not fair: pause so waiting API requests get the lock
        return out

    # ================================================================ search
    def search(self, q: Optional[str] = None, customer: Optional[str] = None, status: Optional[str] = None,
               priority: Optional[int] = None, category: Optional[str] = None, assignee: Optional[int] = None,
               created_from: Optional[int] = None, created_to: Optional[int] = None, limit: int = 50) -> dict:
        """Mini query planner: pick the most selective index, then filter by the remaining predicates.
        Results are newest first."""
        limit = max(1, min(int(limit), 500))
        status = status.upper() if status else None
        if status and status not in STATUSES:
            raise ValidationError(f"unknown status '{status}'")
        cust = customer.strip().lower() if customer else None
        category = category.strip().lower() if category else None
        tokens = tokenize(q) if q else set()
        if q and not tokens:
            raise ValidationError("search text needs words of 2+ letters/digits")

        def match(t: Ticket) -> bool:
            return ((status is None or t.status == status) and
                    (priority is None or t.priority == priority) and
                    (category is None or t.category == category) and
                    (assignee is None or t.assignee == assignee) and
                    (cust is None or t.customer.startswith(cust)) and
                    (created_from is None or t.created_ms >= created_from) and
                    (created_to is None or t.created_ms <= created_to))

        def take(pairs) -> List[Ticket]:
            """Walk an index scan and stop as soon as `limit` matches are found."""
            out: List[Ticket] = []
            for _, t in pairs:
                if match(t):
                    out.append(t)
                    if len(out) >= limit:
                        break
            return out

        any_filter = any(v is not None for v in (status, priority, category, assignee, cust, created_from, created_to))
        t0 = time.perf_counter()
        with self._lock:
            if tokens:
                plan = "inverted index (AND of words) -> filter"
                sets = [self._inverted.get(tok) for tok in tokens]
                if any(s is None for s in sets):
                    found: List[Ticket] = []
                else:
                    sets.sort(key=len)                       # start from the rarest word
                    ids = sets[0] if len(sets) == 1 else set.intersection(*sets)
                    if any_filter:
                        found = take((i, self._by_id.get(i)) for i in sorted(ids, reverse=True))
                    else:
                        found = [self._by_id.get(i) for i in heapq.nlargest(limit, ids)]
            elif cust:
                plan = "B+ tree range scan on (customer, id) prefix -> filter"
                rng = self._by_customer.scan((cust, -1), (cust + _MAX_CHAR, _BIG))
                found = heapq.nlargest(limit, (t for _, t in rng if match(t)), key=lambda t: t.id)
            elif status and created_from is None and created_to is None:
                plan = "B+ tree reverse range scan on (status, id), early stop"
                i = STATUS_INDEX[status]
                found = take(self._by_status.scan((i, -1), (i, _BIG), reverse=True))
            else:
                plan = "B+ tree reverse range scan on (created time, id), early stop"
                lo = (created_from, -1) if created_from is not None else None
                hi = (created_to, _BIG) if created_to is not None else None
                found = take(self._by_time.scan(lo, hi, reverse=True))
            result = [t.to_dict() for t in found]
        return {"plan": plan, "count": len(result), "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
                "tickets": result}

    # ================================================================ views
    def board(self, limit: int = 15) -> dict:
        """Kanban snapshot. NEW column is in true dispatch order (heap peek_n); the other columns are a
        reverse range scan of the (status, id) B+ tree: O(log n + limit), independent of column size."""
        limit = max(1, min(limit, 100))
        with self._lock:
            cols = {}
            for s in STATUSES:
                if s == NEW:
                    ts = [self._by_id.get(tid) for tid, _ in self._queue.peek_n(limit)]
                else:
                    i, ts = STATUS_INDEX[s], []
                    for _, t in self._by_status.scan((i, -1), (i, _BIG), reverse=True):
                        ts.append(t)
                        if len(ts) >= limit:
                            break
                cols[s] = {"count": self._status_count[s], "tickets": [t.to_brief() for t in ts]}
            return cols

    def queue_view(self, limit: int = 20) -> List[dict]:
        with self._lock:
            return [self._by_id.get(tid).to_dict() for tid, _ in self._queue.peek_n(max(1, min(limit, 200)))]

    def stats(self) -> dict:
        with self._lock:
            by_role = {r: 0 for r in ROLES}
            for a in self._agents.values():
                by_role[a.role] += 1
            handlers = [a for a in self._agents.values() if a.role in HANDLER_ROLES]
            return {
                "total_tickets": len(self._by_id),
                "by_status": dict(self._status_count),
                "queue_length": len(self._queue),
                "waiting_by_category": {c: len(h) for c, h in sorted(self._cat_queue.items()) if len(h)},
                "sla_tracked": len(self._sla),
                "agents_total": len(self._agents),
                "agents_by_role": by_role,
                "agents_online": sum(1 for a in handlers if a.online),
                "agents_available": len(self._available),
                "btree_height": {"by_id": self._by_id.height, "by_customer": self._by_customer.height,
                                 "by_time": self._by_time.height, "by_status": self._by_status.height},
                "indexed_tokens": len(self._inverted),
                "db_commits": self.db.commits,
            }

    def dashboard(self, limit: int = 10) -> dict:
        """Everything the live board needs in ONE call."""
        with self._lock:
            return {"stats": self.stats(), "board": self.board(limit), "agents": self.list_agents()}

    # ================================================================ audit (read from PostgreSQL)
    def audit_trail(self, ticket_id: int) -> List[dict]:
        with self._lock:
            self._get(ticket_id)
        return [e._asdict() for e in self.db.get_audit(ticket_id)]

    def ticket_detail(self, ticket_id: int) -> dict:
        """Ticket + its audit trail in ONE call (used by the ticket page)."""
        with self._lock:
            ticket = self._get(ticket_id).to_dict()
        return {"ticket": ticket, "audit": [e._asdict() for e in self.db.get_audit(ticket_id)]}

    def verify_audit(self, ticket_id: int) -> dict:
        with self._lock:
            self._get(ticket_id)
            tail = self._audit_tail.get(ticket_id)
        entries = self.db.get_audit(ticket_id)
        ok, broken = verify_chain(entries)
        if ok and tail and (not entries or (entries[-1].seq, entries[-1].hash) != tail):
            ok, broken = False, (entries[-1].seq + 1 if entries else 1)      # tail rows were removed
        return {"ticket_id": ticket_id, "entries": len(entries), "valid": ok, "first_broken_seq": broken}

    def verify_all_audits(self) -> dict:
        """Stream the whole audit table once (ordered by ticket, seq) and verify every chain: O(total entries)."""
        checked = bad = 0
        cur_id, batch = None, []
        broken: List[int] = []

        def flush():
            nonlocal bad
            if batch:
                ok, _ = verify_chain(batch)
                if not ok:
                    bad += 1
                    if len(broken) < 20:
                        broken.append(batch[0].ticket_id)
        for e in self.db.iter_audit_all():
            if e.ticket_id != cur_id:
                flush()
                cur_id, batch = e.ticket_id, []
            batch.append(e)
            checked += 1
        flush()
        return {"entries_checked": checked, "broken_tickets": bad, "examples": broken, "valid": bad == 0}
