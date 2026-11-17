"""models.py - domain objects, lifecycle state machine, SLA table, agent roles, exceptions."""
from typing import FrozenSet, Optional

# ---------------------------------------------------------------- lifecycle (a small directed graph)
NEW, ASSIGNED, IN_PROGRESS, PENDING, RESOLVED, CLOSED = (
    "NEW", "ASSIGNED", "IN_PROGRESS", "PENDING", "RESOLVED", "CLOSED")
STATUSES = (NEW, ASSIGNED, IN_PROGRESS, PENDING, RESOLVED, CLOSED)
STATUS_INDEX = {s: i for i, s in enumerate(STATUSES)}      # used as the first part of a B+ tree key

# adjacency list -> "is this move legal?" is an O(1) set lookup
TRANSITIONS = {
    NEW:         frozenset({ASSIGNED, CLOSED}),
    ASSIGNED:    frozenset({IN_PROGRESS, NEW}),            # NEW = release back to the queue
    IN_PROGRESS: frozenset({PENDING, RESOLVED}),
    PENDING:     frozenset({IN_PROGRESS, RESOLVED}),       # waiting on customer
    RESOLVED:    frozenset({CLOSED, NEW}),                 # NEW = reopen
    CLOSED:      frozenset(),
}
WORKING = frozenset({ASSIGNED, IN_PROGRESS, PENDING})      # counts toward an agent's load
SLA_RUNNING = frozenset({NEW, ASSIGNED, IN_PROGRESS})      # SLA clock is paused while PENDING

PRIORITY_NAMES = {1: "P1-Critical", 2: "P2-High", 3: "P3-Medium", 4: "P4-Low"}
_H = 3_600_000
SLA_MS = {1: 1 * _H, 2: 4 * _H, 3: 24 * _H, 4: 72 * _H}   # resolution target per priority

CATEGORIES = ("general", "billing", "login", "outage", "shipping", "bug", "feature")
ANY_SKILL = "*"                                            # pool key of a generalist agent

# ---------------------------------------------------------------- agent roles
L1, L2, SUPERVISOR, ADMIN = "L1", "L2", "SUPERVISOR", "ADMIN"
ROLES = (L1, L2, SUPERVISOR, ADMIN)
ROLE_LABELS = {L1: "L1 Agent (front line)", L2: "L2 Senior Agent", SUPERVISOR: "Supervisor", ADMIN: "Admin"}
HANDLER_ROLES = frozenset({L1, L2})                        # only these roles receive and work tickets

# permissions (role -> set of permissions; a check is one O(1) set lookup)
P_WORK_OWN = "work_own_tickets"            # move tickets assigned to you
P_WORK_ANY = "work_any_ticket"             # move any ticket (also close / reopen unassigned ones)
P_PRIORITY_OWN = "change_priority_own"     # re-prioritise tickets assigned to you
P_PRIORITY_ANY = "change_priority_any"
P_ASSIGN = "assign_tickets"                # manual assign / re-assign
P_AVAIL_SELF = "set_own_availability"      # go online / offline yourself
P_AVAIL_ANY = "set_any_availability"
P_ROSTER = "manage_roster"                 # add agents, change role / skills / capacity
P_SYSTEM = "system_operations"             # run dispatcher, SLA sweep, full audit verification

PERMISSIONS = {
    L1:         frozenset({P_WORK_OWN, P_AVAIL_SELF}),
    L2:         frozenset({P_WORK_OWN, P_PRIORITY_OWN, P_AVAIL_SELF}),
    SUPERVISOR: frozenset({P_WORK_ANY, P_PRIORITY_ANY, P_ASSIGN, P_AVAIL_SELF, P_AVAIL_ANY, P_SYSTEM}),
    ADMIN:      frozenset({P_WORK_ANY, P_PRIORITY_ANY, P_ASSIGN, P_AVAIL_SELF, P_AVAIL_ANY, P_SYSTEM, P_ROSTER}),
}
PERMISSION_TEXT = {
    P_WORK_OWN: "move tickets assigned to them", P_WORK_ANY: "move any ticket",
    P_PRIORITY_OWN: "change priority of their own tickets", P_PRIORITY_ANY: "change priority of any ticket",
    P_ASSIGN: "assign or re-assign tickets", P_AVAIL_SELF: "set their own availability",
    P_AVAIL_ANY: "set another agent's availability", P_ROSTER: "manage the agent roster",
    P_SYSTEM: "run system operations",
}
# Routing preference: which tier is tried FIRST for each priority (the other tier is the fallback,
# so a ticket never waits while any suitable agent is free).
TIER_ORDER = {1: (L2, L1), 2: (L2, L1), 3: (L1, L2), 4: (L1, L2)}


# ---------------------------------------------------------------- exceptions
class NotFoundError(Exception):
    pass


class InvalidTransitionError(Exception):
    pass


class CapacityError(Exception):
    pass


class ForbiddenError(Exception):
    pass


class ValidationError(ValueError):
    pass


# ---------------------------------------------------------------- entities
class Ticket:
    __slots__ = ("id", "subject", "description", "customer", "category", "priority", "status",
                 "assignee", "created_ms", "updated_ms", "sla_due_ms", "version", "sla_breached")

    def __init__(self, id: int, subject: str, description: str, customer: str, category: str,
                 priority: int, status: str, assignee: Optional[int], created_ms: int,
                 updated_ms: int, sla_due_ms: int, version: int, sla_breached: bool = False) -> None:
        self.id = id
        self.subject = subject
        self.description = description
        self.customer = customer
        self.category = category
        self.priority = priority
        self.status = status
        self.assignee = assignee
        self.created_ms = created_ms
        self.updated_ms = updated_ms
        self.sla_due_ms = sla_due_ms
        self.version = version
        self.sla_breached = sla_breached

    def as_row(self) -> tuple:
        return (self.id, self.subject, self.description, self.customer, self.category,
                self.priority, self.status, self.assignee, self.created_ms,
                self.updated_ms, self.sla_due_ms, self.version, self.sla_breached)

    def copy(self) -> "Ticket":
        return Ticket(*self.as_row())

    def to_dict(self) -> dict:
        d = {s: getattr(self, s) for s in self.__slots__}
        d["priority_name"] = PRIORITY_NAMES[self.priority]
        return d

    def to_brief(self) -> dict:
        """Small payload for the live board (no description)."""
        return {"id": self.id, "subject": self.subject, "priority": self.priority, "status": self.status,
                "category": self.category, "assignee": self.assignee, "sla_due_ms": self.sla_due_ms,
                "sla_breached": self.sla_breached}


class Agent:
    __slots__ = ("id", "name", "capacity", "online", "created_ms", "role", "skills")

    def __init__(self, id: int, name: str, capacity: int, online: bool, created_ms: int,
                 role: str = L1, skills: FrozenSet[str] = frozenset()) -> None:
        self.id = id
        self.name = name
        self.capacity = capacity
        self.online = online
        self.created_ms = created_ms
        self.role = role
        self.skills = frozenset(skills)          # empty = generalist (handles every category)

    def as_row(self) -> tuple:
        return (self.id, self.name, self.capacity, self.online, self.created_ms, self.role,
                ",".join(sorted(self.skills)))

    def to_dict(self, load: int) -> dict:
        handler = self.role in HANDLER_ROLES
        return {"id": self.id, "name": self.name, "role": self.role, "role_label": ROLE_LABELS[self.role],
                "skills": sorted(self.skills), "takes_tickets": handler,
                "capacity": self.capacity, "online": self.online, "load": load,
                "free_slots": max(0, self.capacity - load) if handler and self.online else 0,
                "created_ms": self.created_ms}
