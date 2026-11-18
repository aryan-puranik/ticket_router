"""
main.py - FastAPI backend.   Run:  uvicorn main:app --port 8000
(Use ONE worker: the data structures live in this process's memory.)

Who is acting?  Staff actions send the header  X-Actor-Id: <agent id>.  The server looks up that
person's role and checks the permission table (models.PERMISSIONS). Customers need no header to
create a ticket, and all read-only endpoints are open.
"""
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from db import Database
from models import (CATEGORIES, HANDLER_ROLES, P_SYSTEM, PERMISSION_TEXT, PERMISSIONS, PRIORITY_NAMES,
                    ROLE_LABELS, ROLES, SLA_MS, STATUSES, TIER_ORDER, TRANSITIONS, CapacityError,
                    ForbiddenError, InvalidTransitionError, NotFoundError, ValidationError)
from service import TicketRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database(config.DB, fast_commit=getattr(config, "FAST_COMMIT", False))
    db.bootstrap()                      # creates the database if it does not exist
    db.init_schema()                    # creates / upgrades tables + append-only trigger
    svc = TicketRouter(db, btree_order=config.BTREE_ORDER)
    svc.load()                          # PostgreSQL -> B+ trees / heaps / hash maps
    svc.start_background(config.BACKGROUND_INTERVAL_S)
    app.state.svc = svc
    yield
    svc.stop_background()
    db.close()


app = FastAPI(title="Customer Support Ticket Router", version="2.0", lifespan=lifespan)


def svc_dep(request: Request) -> TicketRouter:
    return request.app.state.svc


def actor_dep(x_actor_id: Optional[int] = Header(default=None, alias="X-Actor-Id",
                                                  description="Agent id of the staff member doing the action")):
    return x_actor_id


# ------------------------------------------------------------------ error mapping
@app.exception_handler(NotFoundError)
async def _nf(_, e):
    return JSONResponse(status_code=404, content={"detail": str(e)})


@app.exception_handler(ForbiddenError)
async def _fb(_, e):
    return JSONResponse(status_code=403, content={"detail": str(e)})


@app.exception_handler(InvalidTransitionError)
async def _it(_, e):
    return JSONResponse(status_code=409, content={"detail": str(e)})


@app.exception_handler(CapacityError)
async def _ce(_, e):
    return JSONResponse(status_code=409, content={"detail": str(e)})


@app.exception_handler(ValidationError)
async def _ve(_, e):
    return JSONResponse(status_code=422, content={"detail": str(e)})


# ------------------------------------------------------------------ request bodies
class TicketIn(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)
    customer: str = Field(min_length=1, max_length=200)
    category: str = Field(default="general", max_length=50)
    priority: int = Field(default=3, ge=1, le=4)


class BulkIn(BaseModel):
    tickets: List[TicketIn] = Field(min_length=1, max_length=1000)


class TransitionIn(BaseModel):
    to_status: str
    note: str = Field(default="", max_length=1000)


class AssignIn(BaseModel):
    agent_id: int


class PriorityIn(BaseModel):
    priority: int = Field(ge=1, le=4)
    reason: str = Field(default="", max_length=1000)


class AgentIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    capacity: int = Field(default=5, ge=1, le=100)
    role: str = "L1"
    skills: List[str] = []


class AgentPatch(BaseModel):
    online: Optional[bool] = None
    capacity: Optional[int] = Field(default=None, ge=1, le=100)
    role: Optional[str] = None
    skills: Optional[List[str]] = None


# ------------------------------------------------------------------ meta / health
@app.get("/health")
def health(svc: TicketRouter = Depends(svc_dep)):
    return {"status": "ok", **svc.stats()}


@app.get("/meta")
def meta():
    return {"statuses": list(STATUSES),
            "transitions": {s: sorted(v) for s, v in TRANSITIONS.items()},
            "priorities": {str(k): v for k, v in PRIORITY_NAMES.items()},
            "sla_hours": {str(k): v / 3_600_000 for k, v in SLA_MS.items()},
            "categories": list(CATEGORIES),
            "roles": [{"role": r, "label": ROLE_LABELS[r], "takes_tickets": r in HANDLER_ROLES,
                       "permissions": sorted(PERMISSIONS[r])} for r in ROLES],
            "permission_text": PERMISSION_TEXT,
            "tier_order": {str(p): list(v) for p, v in TIER_ORDER.items()}}


@app.get("/stats")
def stats(svc: TicketRouter = Depends(svc_dep)):
    return svc.stats()


@app.get("/dashboard")
def dashboard(limit: int = Query(10, ge=1, le=100), svc: TicketRouter = Depends(svc_dep)):
    return svc.dashboard(limit)


# ------------------------------------------------------------------ agents
@app.post("/agents", status_code=201)
def create_agent(body: AgentIn, svc: TicketRouter = Depends(svc_dep), actor: Optional[int] = Depends(actor_dep)):
    return svc.create_agent(body.name, body.capacity, body.role, body.skills, actor)


@app.get("/agents")
def list_agents(svc: TicketRouter = Depends(svc_dep)):
    return svc.list_agents()


@app.patch("/agents/{agent_id}")
def patch_agent(agent_id: int, body: AgentPatch, svc: TicketRouter = Depends(svc_dep),
                actor: Optional[int] = Depends(actor_dep)):
    return svc.update_agent(agent_id, actor, body.online, body.capacity, body.role, body.skills)


# ------------------------------------------------------------------ tickets
@app.post("/tickets", status_code=201)
def create_ticket(body: TicketIn, svc: TicketRouter = Depends(svc_dep)):
    return svc.create_ticket(body.subject, body.description, body.customer, body.category, body.priority)


@app.post("/tickets/bulk", status_code=201)
def create_bulk(body: BulkIn, svc: TicketRouter = Depends(svc_dep)):
    return svc.create_tickets_bulk([t.model_dump() for t in body.tickets])


@app.get("/tickets")
def search_tickets(q: Optional[str] = None, customer: Optional[str] = None, status: Optional[str] = None,
                   priority: Optional[int] = Query(None, ge=1, le=4), category: Optional[str] = None,
                   assignee: Optional[int] = None, created_from: Optional[int] = None,
                   created_to: Optional[int] = None, limit: int = Query(50, ge=1, le=500),
                   svc: TicketRouter = Depends(svc_dep)):
    return svc.search(q, customer, status, priority, category, assignee, created_from, created_to, limit)


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int, svc: TicketRouter = Depends(svc_dep)):
    return svc.get_ticket(ticket_id)


@app.get("/tickets/{ticket_id}/detail")
def ticket_detail(ticket_id: int, svc: TicketRouter = Depends(svc_dep)):
    return svc.ticket_detail(ticket_id)


@app.post("/tickets/{ticket_id}/transition")
def transition(ticket_id: int, body: TransitionIn, svc: TicketRouter = Depends(svc_dep),
               actor: Optional[int] = Depends(actor_dep)):
    return svc.transition(ticket_id, body.to_status, actor, body.note)


@app.post("/tickets/{ticket_id}/assign")
def assign(ticket_id: int, body: AssignIn, svc: TicketRouter = Depends(svc_dep),
           actor: Optional[int] = Depends(actor_dep)):
    return svc.assign(ticket_id, body.agent_id, actor)


@app.post("/tickets/{ticket_id}/priority")
def priority(ticket_id: int, body: PriorityIn, svc: TicketRouter = Depends(svc_dep),
             actor: Optional[int] = Depends(actor_dep)):
    return svc.change_priority(ticket_id, body.priority, actor, body.reason)


@app.get("/tickets/{ticket_id}/audit")
def audit(ticket_id: int, svc: TicketRouter = Depends(svc_dep)):
    return svc.audit_trail(ticket_id)


@app.get("/tickets/{ticket_id}/audit/verify")
def verify_audit(ticket_id: int, svc: TicketRouter = Depends(svc_dep)):
    return svc.verify_audit(ticket_id)


# ------------------------------------------------------------------ live views / operations
@app.get("/board")
def board(limit: int = Query(15, ge=1, le=100), svc: TicketRouter = Depends(svc_dep)):
    return svc.board(limit)


@app.get("/queue")
def queue(limit: int = Query(20, ge=1, le=200), svc: TicketRouter = Depends(svc_dep)):
    return svc.queue_view(limit)


@app.post("/dispatch")
def dispatch(svc: TicketRouter = Depends(svc_dep), actor: Optional[int] = Depends(actor_dep)):
    svc.check(actor, P_SYSTEM)
    return svc.dispatch()


@app.post("/sla/escalate")
def escalate(svc: TicketRouter = Depends(svc_dep), actor: Optional[int] = Depends(actor_dep)):
    svc.check(actor, P_SYSTEM)
    return svc.escalate_breached()


@app.get("/audit/verify-all")
def verify_all(svc: TicketRouter = Depends(svc_dep), actor: Optional[int] = Depends(actor_dep)):
    svc.check(actor, P_SYSTEM)
    return svc.verify_all_audits()
