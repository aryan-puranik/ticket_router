"""Settings. Edit PGPASSWORD below (or set environment variables) to match YOUR PostgreSQL install."""
import os

DB = dict(
    host=os.getenv("PGHOST", "127.0.0.1"),            # 127.0.0.1, not "localhost" (see README: Windows delay)
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "ticket_router"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "Aryan22"),     # <-- change to the password you set during Postgres install
)

BTREE_ORDER = 64               # max children per B+ tree node
BACKGROUND_INTERVAL_S = 5      # how often the SLA-escalation + dispatch worker runs
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Optional turbo switch. False (default) = every commit is flushed to disk before the API answers.
# True = PostgreSQL answers before the flush (synchronous_commit=off): much faster on slow disks;
# the database can never be corrupted by it, but a power cut may lose the last ~0.5 s of changes.
FAST_COMMIT = os.getenv("FAST_COMMIT", "0") == "1"
