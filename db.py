"""
db.py - PostgreSQL persistence layer (psycopg2, raw SQL).

Architecture: PostgreSQL is the durable source of truth. At start-up the service streams the
tables into memory and builds the B+ trees / heaps (service.py). Every API request writes ALL
of its changes (ticket rows + audit rows) in ONE transaction = one commit = one disk flush.
Rows are sent in bulk (execute_values), so 500 changes cost about the same as 1.
"""
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2 import sql

from audit import AuditEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id          BIGINT PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,
    capacity    INT     NOT NULL CHECK (capacity > 0),
    online      BOOLEAN NOT NULL DEFAULT TRUE,
    created_ms  BIGINT  NOT NULL
);
-- roles were added in v2: these lines upgrade an existing v1 database in place
ALTER TABLE agents ADD COLUMN IF NOT EXISTS role   TEXT NOT NULL DEFAULT 'L1';
ALTER TABLE agents ADD COLUMN IF NOT EXISTS skills TEXT NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS tickets (
    id          BIGINT PRIMARY KEY,
    subject     TEXT     NOT NULL,
    description TEXT     NOT NULL DEFAULT '',
    customer    TEXT     NOT NULL,
    category    TEXT     NOT NULL,
    priority    SMALLINT NOT NULL CHECK (priority BETWEEN 1 AND 4),
    status      TEXT     NOT NULL,
    assignee    BIGINT   REFERENCES agents(id),
    created_ms  BIGINT   NOT NULL,
    updated_ms  BIGINT   NOT NULL,
    sla_due_ms  BIGINT   NOT NULL,
    version     INT      NOT NULL
);
ALTER TABLE tickets ADD COLUMN IF NOT EXISTS sla_breached BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS ticket_audit (
    ticket_id   BIGINT NOT NULL REFERENCES tickets(id),
    seq         INT    NOT NULL,
    ts_ms       BIGINT NOT NULL,
    actor       TEXT   NOT NULL,
    action      TEXT   NOT NULL,
    detail      TEXT   NOT NULL,
    prev_hash   CHAR(64) NOT NULL,
    hash        CHAR(64) NOT NULL,
    PRIMARY KEY (ticket_id, seq)
);

-- compliance: the audit table is append-only, even for a careless admin
CREATE OR REPLACE FUNCTION audit_is_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'ticket_audit is append-only (% blocked)', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_append_only ON ticket_audit;
CREATE TRIGGER trg_audit_append_only
    BEFORE UPDATE OR DELETE ON ticket_audit
    FOR EACH ROW EXECUTE FUNCTION audit_is_append_only();
"""

TICKET_COLS = ("id, subject, description, customer, category, priority, status, "
               "assignee, created_ms, updated_ms, sla_due_ms, version, sla_breached")
AGENT_COLS = "id, name, capacity, online, created_ms, role, skills"
AUDIT_COLS = "ticket_id, seq, ts_ms, actor, action, detail, prev_hash, hash"

UPSERT_TICKETS = f"""
INSERT INTO tickets ({TICKET_COLS}) VALUES %s
ON CONFLICT (id) DO UPDATE SET
    priority = EXCLUDED.priority, status = EXCLUDED.status, assignee = EXCLUDED.assignee,
    updated_ms = EXCLUDED.updated_ms, sla_due_ms = EXCLUDED.sla_due_ms, version = EXCLUDED.version,
    sla_breached = EXCLUDED.sla_breached
"""
INSERT_AUDITS = f"INSERT INTO ticket_audit ({AUDIT_COLS}) VALUES %s"
UPSERT_AGENTS = f"""
INSERT INTO agents ({AGENT_COLS}) VALUES %s
ON CONFLICT (id) DO UPDATE SET capacity = EXCLUDED.capacity, online = EXCLUDED.online,
    role = EXCLUDED.role, skills = EXCLUDED.skills
"""
_COLS = {"agents": AGENT_COLS, "tickets": TICKET_COLS, "ticket_audit": AUDIT_COLS}


def _copy_text(value) -> str:
    """One value in PostgreSQL COPY text format."""
    if value is None:
        return "\\N"
    if value is True:
        return "t"
    if value is False:
        return "f"
    if isinstance(value, str):
        return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    return str(value)


class _RowStream:
    """File-like object that feeds rows to COPY in chunks (no giant string in memory)."""

    def __init__(self, rows: Iterable[tuple], chunk_rows: int = 5000) -> None:
        self._it = iter(rows)
        self._chunk = chunk_rows

    def read(self, _size: int = -1) -> str:
        out = []
        for row in self._it:
            out.append("\t".join(map(_copy_text, row)))
            if len(out) >= self._chunk:
                break
        return "\n".join(out) + "\n" if out else ""

    readline = read


class Database:
    def __init__(self, cfg: dict, minconn: int = 1, maxconn: int = 10, fast_commit: bool = False) -> None:
        self.cfg = dict(cfg)
        if fast_commit:      # answer before the WAL flush; consistent, but the last moments can be lost in a crash
            self.cfg["options"] = (self.cfg.get("options", "") + " -c synchronous_commit=off").strip()
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._minconn, self._maxconn = minconn, maxconn
        self.commits = 0                    # counter used by tests / benchmarks

    # ---------------------------------------------------------------- setup
    def bootstrap(self) -> None:
        """Create the database itself if it does not exist yet (connects to the 'postgres' DB)."""
        cfg = dict(self.cfg, dbname="postgres")
        conn = psycopg2.connect(**cfg)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.cfg["dbname"],))
                if cur.fetchone() is None:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.cfg["dbname"])))
        finally:
            conn.close()

    def connect(self) -> None:
        if self._pool is None:
            self._pool = psycopg2.pool.ThreadedConnectionPool(self._minconn, self._maxconn, **self.cfg)

    def init_schema(self) -> None:
        self.connect()
        with self.tx() as cur:
            cur.execute(SCHEMA)

    def reset(self) -> None:
        """Wipe all data (TRUNCATE does not fire the append-only row trigger)."""
        with self.tx() as cur:
            cur.execute("TRUNCATE ticket_audit, tickets, agents")

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    @contextmanager
    def tx(self):
        """One transaction: commit on success, rollback on any exception."""
        self.connect()
        conn = self._pool.getconn()
        try:
            with conn:                       # psycopg2: commits / rolls back
                with conn.cursor() as cur:
                    yield cur
            self.commits += 1
        finally:
            self._pool.putconn(conn)

    # ---------------------------------------------------------------- bulk writes (used inside tx())
    @staticmethod
    def upsert_tickets(cur, tickets: Iterable) -> None:
        psycopg2.extras.execute_values(cur, UPSERT_TICKETS, [t.as_row() for t in tickets], page_size=1000)

    @staticmethod
    def insert_audits(cur, entries: List[AuditEntry]) -> None:
        psycopg2.extras.execute_values(cur, INSERT_AUDITS, entries, page_size=1000)

    @staticmethod
    def upsert_agents(cur, agents: Iterable) -> None:
        psycopg2.extras.execute_values(cur, UPSERT_AGENTS, [a.as_row() for a in agents], page_size=1000)

    @staticmethod
    def copy_rows(cur, table: str, rows: Iterable[tuple]) -> None:
        """Fastest way to load many rows (PostgreSQL COPY). Used by seed.py."""
        cur.copy_expert(f"COPY {table} ({_COLS[table]}) FROM STDIN", _RowStream(rows))

    # ---------------------------------------------------------------- reads
    def load_agents(self) -> List[tuple]:
        with self.tx() as cur:
            cur.execute(f"SELECT {AGENT_COLS} FROM agents ORDER BY id")
            return cur.fetchall()

    def iter_tickets(self, itersize: int = 20000) -> Iterator[tuple]:
        """Stream tickets ordered by id using a server-side cursor (constant client memory)."""
        self.connect()
        conn = self._pool.getconn()
        try:
            with conn.cursor(name="load_tickets") as cur:
                cur.itersize = itersize
                cur.execute(f"SELECT {TICKET_COLS} FROM tickets ORDER BY id")
                for row in cur:
                    yield row
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def load_audit_tails(self) -> List[Tuple[int, int, str]]:
        """(ticket_id, last_seq, last_hash) per ticket - all the service needs in memory."""
        with self.tx() as cur:
            cur.execute("SELECT DISTINCT ON (ticket_id) ticket_id, seq, hash "
                        "FROM ticket_audit ORDER BY ticket_id, seq DESC")
            return cur.fetchall()

    def get_audit(self, ticket_id: int) -> List[AuditEntry]:
        with self.tx() as cur:
            cur.execute(f"SELECT {AUDIT_COLS} FROM ticket_audit WHERE ticket_id = %s ORDER BY seq", (ticket_id,))
            return [AuditEntry(*r) for r in cur.fetchall()]

    def iter_audit_all(self, itersize: int = 20000) -> Iterator[AuditEntry]:
        self.connect()
        conn = self._pool.getconn()
        try:
            with conn.cursor(name="verify_audit") as cur:
                cur.itersize = itersize
                cur.execute(f"SELECT {AUDIT_COLS} FROM ticket_audit ORDER BY ticket_id, seq")
                for r in cur:
                    yield AuditEntry(*r)
            conn.commit()
        finally:
            self._pool.putconn(conn)
