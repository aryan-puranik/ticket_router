"""
audit.py - Tamper-evident audit trail (hash chain).

Every ticket revision is an append-only entry:

    hash_n = SHA256( hash_(n-1), ticket_id, seq, timestamp, actor, action, detail )

It is a linked list whose "pointer" is the previous entry's hash. Editing or deleting any
old entry breaks every hash after it, so verification is O(k) for k entries of one ticket.
(PostgreSQL additionally blocks UPDATE/DELETE on the table with a trigger - see db.py.)
"""
import hashlib
import json
from typing import Iterable, NamedTuple, Optional, Tuple

GENESIS = "0" * 64


class AuditEntry(NamedTuple):
    ticket_id: int
    seq: int
    ts_ms: int
    actor: str
    action: str
    detail: str        # canonical JSON string
    prev_hash: str
    hash: str


def canonical(detail: dict) -> str:
    return json.dumps(detail, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(prev_hash: str, ticket_id: int, seq: int, ts_ms: int,
                 actor: str, action: str, detail: str) -> str:
    payload = json.dumps([prev_hash, ticket_id, seq, ts_ms, actor, action, detail],
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_entry(prev: Optional[Tuple[int, str]], ticket_id: int, ts_ms: int,
               actor: str, action: str, detail: dict) -> AuditEntry:
    """prev = (last_seq, last_hash) of this ticket's chain, or None for the first entry."""
    seq, prev_hash = (prev[0] + 1, prev[1]) if prev else (1, GENESIS)
    d = canonical(detail)
    return AuditEntry(ticket_id, seq, ts_ms, actor, action, d, prev_hash,
                      compute_hash(prev_hash, ticket_id, seq, ts_ms, actor, action, d))


def verify_chain(entries: Iterable[AuditEntry]) -> Tuple[bool, Optional[int]]:
    """Returns (is_valid, seq_of_first_broken_entry)."""
    prev_hash, expected_seq = GENESIS, 1
    for e in entries:
        if e.seq != expected_seq or e.prev_hash != prev_hash:
            return False, e.seq
        if compute_hash(e.prev_hash, e.ticket_id, e.seq, e.ts_ms, e.actor, e.action, e.detail) != e.hash:
            return False, e.seq
        prev_hash, expected_seq = e.hash, expected_seq + 1
    return True, None
