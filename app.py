"""
app.py - Streamlit front end.   Run:  streamlit run app.py
Talks to the FastAPI backend over HTTP (default http://127.0.0.1:8000).

Why it is fast:
  * one page is rendered at a time (sidebar menu), so a click costs 1 API call, not 8-9
  * one keep-alive HTTP session is reused (no new connection per call)
  * 127.0.0.1 is used instead of "localhost" (on Windows "localhost" can add ~2 s to EVERY call)
  * combined endpoints: /dashboard for the live board, /tickets/{id}/detail for the ticket page
"""
import json
import random
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

import config

st.set_page_config(page_title="Support Ticket Router", page_icon="🎫", layout="wide")

STATUS_ORDER = ["NEW", "ASSIGNED", "IN_PROGRESS", "PENDING", "RESOLVED", "CLOSED"]
SLA_RUNNING = {"NEW", "ASSIGNED", "IN_PROGRESS"}
WORKING = {"ASSIGNED", "IN_PROGRESS", "PENDING"}
PRI = {1: "P1-Critical", 2: "P2-High", 3: "P3-Medium", 4: "P4-Low"}
PAGES = ["📊 Live board", "➕ New ticket", "🔎 Search", "🧾 Ticket & audit", "👥 Agents & roles", "⚙️ System"]


def fast_url(url: str) -> str:
    return url.rstrip("/").replace("//localhost", "//127.0.0.1")


# ------------------------------------------------------------------ HTTP
@st.cache_resource
def http() -> requests.Session:
    s = requests.Session()                    # keep-alive: connections are reused across reruns
    s.mount("http://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8))
    return s


def call(method: str, path: str, **kw):
    headers = {}
    if st.session_state.get("actor_id"):
        headers["X-Actor-Id"] = str(st.session_state["actor_id"])
    t0 = time.perf_counter()
    try:
        r = http().request(method, API + path, timeout=60, headers=headers, **kw)
    except requests.RequestException as e:
        return False, f"Cannot reach the API at {API} ({e.__class__.__name__}). Is `uvicorn main:app --port 8000` running?"
    st.session_state["last_ms"] = (time.perf_counter() - t0) * 1000
    try:
        data = r.json()
    except ValueError:
        data = r.text
    if r.status_code >= 400:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        if isinstance(detail, list):
            detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
        return False, str(detail)
    return True, data


@st.cache_data(ttl=600, show_spinner=False)
def get_meta(api: str) -> dict:
    ok, d = call("GET", "/meta")
    if not ok:
        raise RuntimeError(d)
    return d


@st.cache_data(ttl=20, show_spinner=False)
def get_staff(api: str) -> list:
    ok, d = call("GET", "/agents")
    if not ok:
        raise RuntimeError(d)
    return d


# ------------------------------------------------------------------ small UI helpers
def show_df(df: pd.DataFrame, height=None):
    kw = {"hide_index": True}
    if height is not None:
        kw["height"] = int(height)
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:                      # older Streamlit versions
        st.dataframe(df, use_container_width=True, **kw)


def wide_button(where, label: str, key: str, **kw) -> bool:
    try:
        return where.button(label, key=key, width="stretch", **kw)
    except TypeError:                      # older Streamlit versions
        return where.button(label, key=key, use_container_width=True, **kw)


def fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000).strftime("%d %b %H:%M:%S")


def sla_text(t: dict) -> str:
    if t["status"] not in SLA_RUNNING:
        return "paused" if t["status"] == "PENDING" else "-"
    if t.get("sla_breached"):
        return "BREACHED"
    mins = int((t["sla_due_ms"] - time.time() * 1000) / 60000)
    h, m = divmod(abs(mins), 60)
    return f"OVERDUE {h}h{m:02d}m" if mins < 0 else f"{h}h{m:02d}m left"


def flash(kind: str, msg: str) -> None:
    st.session_state["flash"] = (kind, msg)


def act(method: str, path: str, ok_msg: str, roster: bool = False, **kw) -> None:
    ok, res = call(method, path, **kw)
    if ok and roster:
        get_staff.clear()
    flash("success" if ok else "error", ok_msg if ok else str(res))
    st.rerun()


def ticket_rows(tickets, names):
    return [{"#": t["id"], "Pri": f"P{t['priority']}", "Subject": t["subject"][:40], "Type": t["category"],
             "SLA": sla_text(t), "Agent": names.get(t["assignee"], "")} for t in tickets]


# ------------------------------------------------------------------ sidebar: connection, sign-in, menu
st.sidebar.title("🎫 Ticket Router")
API = fast_url(st.session_state.get("api_url", config.API_URL))
try:
    META = get_meta(API)
    STAFF = get_staff(API)
except RuntimeError as e:
    st.error(str(e))
    st.stop()
ROLE_PERMS = {r["role"]: set(r["permissions"]) for r in META["roles"]}
ROLE_LABEL = {r["role"]: r["label"] for r in META["roles"]}
NAMES = {a["id"]: a["name"] for a in STAFF}

ids = [a["id"] for a in STAFF]
default = next((a["id"] for a in STAFF if a["role"] == "ADMIN"), ids[0])
if st.session_state.get("actor_id") not in ids:
    st.session_state["actor_id"] = default
st.sidebar.selectbox("Signed in as", ids, key="actor_id",
                     format_func=lambda i: next(f"{a['name']} · {ROLE_LABEL[a['role']]}" for a in STAFF if a["id"] == i),
                     help="Every action is checked against this person's role on the server and written to the audit trail.")
ME = next(a for a in STAFF if a["id"] == st.session_state["actor_id"])
MY_PERMS = ROLE_PERMS[ME["role"]]
page = st.sidebar.radio("Menu", PAGES, key="page")
with st.sidebar.expander("Settings"):
    LIMIT = st.slider("Tickets per board column", 5, 30, 10)
    AUTO = st.toggle("Auto-refresh board (3 s)", True)
    st.text_input("API URL", config.API_URL, key="api_url")
st.sidebar.caption("B+ trees · indexed heaps · hash maps · hash-chained audit log")


def can(perm: str) -> bool:
    return perm in MY_PERMS


def can_ticket(t: dict, own: str, any_: str) -> bool:
    return any_ in MY_PERMS or (own in MY_PERMS and t["assignee"] == ME["id"])


if "flash" in st.session_state:
    kind, msg = st.session_state.pop("flash")
    getattr(st, kind)(msg)


# ------------------------------------------------------------------ live board (1 API call per refresh)
def live_board(limit: int):
    ok, d = call("GET", "/dashboard", params={"limit": limit})
    if not ok:
        st.error(d)
        return
    stats, board, agents = d["stats"], d["board"], d["agents"]
    names = {a["id"]: a["name"] for a in agents}
    cols = st.columns(7)
    cols[0].metric("Total", f"{stats['total_tickets']:,}")
    for c, s in zip(cols[1:], STATUS_ORDER):
        c.metric(s.replace("_", " ").title(), f"{board[s]['count']:,}")
    waiting = ", ".join(f"{c} {n:,}" for c, n in stats["waiting_by_category"].items()) or "none"
    st.caption(f"Waiting by type: {waiting}  ·  agents with a free slot: {stats['agents_available']}"
               f"/{stats['agents_online']} online  ·  refreshed {time.strftime('%H:%M:%S')}"
               f" in {st.session_state.get('last_ms', 0):.0f} ms")
    for c, s in zip(st.columns(4), ["NEW", "ASSIGNED", "IN_PROGRESS", "PENDING"]):
        with c:
            st.markdown(f"**{s.replace('_', ' ')}**" + ("  _(next to be dispatched first)_" if s == "NEW" else ""))
            rows = ticket_rows(board[s]["tickets"], names)
            if rows:
                show_df(pd.DataFrame(rows), height=38 + 35 * len(rows))
            else:
                st.caption("empty")
    with st.expander("Resolved & Closed (latest)"):
        for c, s in zip(st.columns(2), ("RESOLVED", "CLOSED")):
            with c:
                st.markdown(f"**{s}**")
                rows = ticket_rows(board[s]["tickets"], names)
                if rows:
                    show_df(pd.DataFrame(rows))
    with st.expander("Team workload"):
        team = [{"Agent": a["name"], "Role": a["role"], "Skills": ", ".join(a["skills"]) or "all (generalist)",
                 "Online": a["online"], "Load": f"{a['load']}/{a['capacity']}"} for a in agents if a["takes_tickets"]]
        if team:
            show_df(pd.DataFrame(team))


# ------------------------------------------------------------------ ticket detail + actions + audit (1 API call)
def ticket_panel(tid: int, kp: str) -> None:
    ok, d = call("GET", f"/tickets/{tid}/detail")
    if not ok:
        st.error(d)
        return
    t, trail = d["ticket"], d["audit"]
    c = st.columns(5)
    c[0].metric("Ticket", f"#{t['id']}")
    c[1].metric("Status", t["status"])
    c[2].metric("Priority", PRI[t["priority"]])
    c[3].metric("Assignee", NAMES.get(t["assignee"], "-"))
    c[4].metric("SLA", sla_text(t))
    st.markdown(f"**{t['subject']}**  ·  {t['customer']}  ·  {t['category']}  ·  version {t['version']}")
    if t["description"]:
        st.write(t["description"])

    st.markdown("##### Move ticket")
    nexts = [s for s in META["transitions"].get(t["status"], []) if s != "ASSIGNED"]
    if not nexts:
        st.caption("This ticket is CLOSED (terminal state).")
    elif not can_ticket(t, "work_own_tickets", "work_any_ticket"):
        st.info(f"You are signed in as {ME['name']} ({ROLE_LABEL[ME['role']]}). "
                + ("This ticket is not assigned to you, so you cannot move it."
                   if "work_own_tickets" in MY_PERMS else "Your role cannot move tickets."))
    else:
        note = st.text_input("Note (optional, saved in audit trail)", key=f"{kp}_note")
        for col, s in zip(st.columns(len(nexts)), nexts):
            if wide_button(col, f"→ {s}", key=f"{kp}_to_{s}"):
                act("POST", f"/tickets/{tid}/transition", f"Ticket #{tid} moved to {s}",
                    json={"to_status": s, "note": note})

    if t["status"] == "NEW" or t["status"] in WORKING:
        if can("assign_tickets"):
            with st.expander("Assign / re-assign"):
                free = [a for a in STAFF if a["takes_tickets"] and a["online"] and a["id"] != t["assignee"]]
                if free:
                    by_id = {a["id"]: a for a in free}
                    pick = st.selectbox("Agent", list(by_id), key=f"{kp}_agent", format_func=lambda i: (
                        f"{by_id[i]['name']} · {by_id[i]['role']} · {', '.join(by_id[i]['skills']) or 'generalist'}"))
                    if st.button("Assign", key=f"{kp}_assign"):
                        act("POST", f"/tickets/{tid}/assign", f"Ticket #{tid} assigned to {by_id[pick]['name']}",
                            roster=True, json={"agent_id": pick})
                else:
                    st.caption("No other agent is online.")
        if can_ticket(t, "change_priority_own", "change_priority_any"):
            with st.expander("Change priority"):
                p = st.selectbox("New priority", [1, 2, 3, 4], index=t["priority"] - 1, key=f"{kp}_pri",
                                 format_func=lambda x: PRI[x])
                why = st.text_input("Reason", key=f"{kp}_why")
                if st.button("Update priority", key=f"{kp}_setpri"):
                    act("POST", f"/tickets/{tid}/priority", f"Priority of #{tid} set to {PRI[p]}",
                        json={"priority": p, "reason": why})

    st.markdown("##### Audit trail (tamper-evident, append-only)")
    rows = []
    for e in trail:
        det = json.loads(e["detail"])
        ch = "; ".join(f"{k}: {v[0]} → {v[1]}" for k, v in det.get("changes", {}).items())
        info = ", ".join(f"{k}={v}" for k, v in det.items() if k not in ("changes", "version", "actor_role"))
        who = e["actor"] + (f" ({det['actor_role']})" if det.get("actor_role") else "")
        rows.append({"#": e["seq"], "Time": fmt_time(e["ts_ms"]), "Actor": who, "Action": e["action"],
                     "Changes": ch, "Info": info, "Hash": e["hash"][:12] + "…"})
    show_df(pd.DataFrame(rows))
    if st.button("Verify integrity of this ticket's chain", key=f"{kp}_verify"):
        ok, r = call("GET", f"/tickets/{tid}/audit/verify")
        if ok and r["valid"]:
            st.success(f"Chain valid - {r['entries']} entries, no tampering detected.")
        else:
            st.error(f"Chain BROKEN: {r}")


# ------------------------------------------------------------------ pages
def page_new():
    with st.form("new_ticket", clear_on_submit=True):
        subject = st.text_input("Subject *", max_chars=200)
        desc = st.text_area("Description")
        c1, c2, c3 = st.columns(3)
        customer = c1.text_input("Customer email *")
        category = c2.selectbox("Category", META["categories"])
        priority = c3.selectbox("Priority", [1, 2, 3, 4], index=2, format_func=lambda p: PRI[p])
        if st.form_submit_button("Create ticket", type="primary"):
            ok, res = call("POST", "/tickets", json={"subject": subject, "description": desc, "customer": customer,
                                                     "category": category, "priority": priority})
            if ok:
                st.success(f"Ticket #{res['id']} created in {st.session_state['last_ms']:.0f} ms → status {res['status']}"
                           + (f", auto-assigned to {NAMES.get(res['assignee'], 'agent #%s' % res['assignee'])}"
                              if res["assignee"] else ", waiting in queue"))
            else:
                st.error(res)


def page_search():
    with st.form("search"):
        c1, c2, c3 = st.columns(3)
        q = c1.text_input("Free text (all words must match)", placeholder="refund payment")
        cust = c2.text_input("Customer (prefix ok)", placeholder="user12")
        cat = c3.selectbox("Category", ["(any)"] + META["categories"])
        c4, c5, c6, c7 = st.columns(4)
        status = c4.selectbox("Status", ["(any)"] + STATUS_ORDER)
        pri = c5.selectbox("Priority", ["(any)", 1, 2, 3, 4])
        since = c6.selectbox("Created", ["any time", "last hour", "last 24 hours", "last 7 days", "last 30 days"])
        limit = c7.number_input("Max results", 1, 500, 50)
        mine = st.checkbox("Only tickets assigned to me")
        go = st.form_submit_button("Search", type="primary")
    if go:
        p = {"limit": int(limit)}
        if q.strip():
            p["q"] = q.strip()
        if cust.strip():
            p["customer"] = cust.strip()
        if cat != "(any)":
            p["category"] = cat
        if status != "(any)":
            p["status"] = status
        if pri != "(any)":
            p["priority"] = pri
        if mine:
            p["assignee"] = ME["id"]
        hours = {"last hour": 1, "last 24 hours": 24, "last 7 days": 168, "last 30 days": 720}.get(since)
        if hours:
            p["created_from"] = int(time.time() * 1000) - hours * 3_600_000
        ok, res = call("GET", "/tickets", params=p)
        st.session_state["search_res"] = res if ok else {"error": res}
    res = st.session_state.get("search_res")
    if res:
        if "error" in res:
            st.error(res["error"])
            return
        st.caption(f"**{res['count']}** result(s) in **{res['elapsed_ms']} ms**  ·  query plan: _{res['plan']}_")
        if res["tickets"]:
            df = pd.DataFrame([{"#": t["id"], "Status": t["status"], "Pri": f"P{t['priority']}", "Subject": t["subject"],
                                "Customer": t["customer"], "Category": t["category"],
                                "Agent": NAMES.get(t["assignee"], ""), "Created": fmt_time(t["created_ms"])}
                               for t in res["tickets"]])
            show_df(df, height=min(38 + 35 * len(df), 420))
            pick = st.selectbox("Open ticket", ["(none)"] + [t["id"] for t in res["tickets"]], key="search_open")
            if pick != "(none)":
                with st.container(border=True):
                    ticket_panel(pick, "s")


def page_ticket():
    tid = st.number_input("Ticket #", min_value=1, value=1, step=1, key="tid")
    ticket_panel(int(tid), "t")


def page_agents():
    ok, agents = call("GET", "/agents")
    if not ok:
        st.error(agents)
        return
    df = pd.DataFrame([{"ID": a["id"], "Name": a["name"], "Role": a["role_label"],
                        "Skills": (", ".join(a["skills"]) or "all (generalist)") if a["takes_tickets"] else "-",
                        "Online": a["online"] if a["takes_tickets"] else None,
                        "Load": f"{a['load']}/{a['capacity']}" if a["takes_tickets"] else "-"} for a in agents])
    show_df(df, height=min(38 + 35 * len(df), 460))
    with st.expander("What each role may do (enforced by the server)"):
        text = META["permission_text"]
        show_df(pd.DataFrame([{"Permission": text[p].replace("their", "your").replace("them", "you").capitalize(),
                               **{r["label"]: "✔" if p in r["permissions"] else "" for r in META["roles"]}}
                              for p in text]))
        st.caption("Routing: P1/P2 tickets try an L2 senior agent first, P3/P4 try an L1 agent first; the other tier is "
                   "the fallback. A specialist only receives tickets of their skills; a generalist receives any type. "
                   "Supervisors and Admins manage the desk and do not receive tickets.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Add staff member**")
        if not can("manage_roster"):
            st.caption("Only an Admin can add staff or change roles, skills and capacity.")
        else:
            with st.form("new_agent", clear_on_submit=True):
                n = st.text_input("Name")
                role = st.selectbox("Role", [r["role"] for r in META["roles"]], format_func=lambda r: ROLE_LABEL[r])
                sk = st.multiselect("Skills (leave empty = generalist)", META["categories"])
                cap = st.number_input("Capacity (tickets at once)", 1, 100, 5)
                if st.form_submit_button("Add"):
                    ok2, r = call("POST", "/agents", json={"name": n, "capacity": int(cap), "role": role, "skills": sk})
                    if ok2:
                        get_staff.clear()
                        flash("success", f"{r['name']} added as {r['role_label']}")
                        st.rerun()
                    st.error(r)
    with c2:
        st.markdown("**Manage staff member**")
        by_id = {x["id"]: x for x in agents}
        a = by_id[st.selectbox("Person", list(by_id), key="manage_who",
                               format_func=lambda i: f"{by_id[i]['name']} · {by_id[i]['role']}")]
        may_toggle = can("set_any_availability") or (can("set_own_availability") and a["id"] == ME["id"])
        if a["takes_tickets"] and may_toggle:
            if st.button("Go offline" if a["online"] else "Go online", key="toggle_online"):
                act("PATCH", f"/agents/{a['id']}", f"{a['name']} is now {'offline' if a['online'] else 'online'}",
                    roster=True, json={"online": not a["online"]})
            st.caption("Going offline returns not-yet-started tickets to the queue.")
        if can("manage_roster"):
            roles = [r["role"] for r in META["roles"]]
            nrole = st.selectbox("Role", roles, index=roles.index(a["role"]), key=f"role{a['id']}",
                                 format_func=lambda r: ROLE_LABEL[r])
            nsk = st.multiselect("Skills (empty = generalist)", META["categories"], default=a["skills"], key=f"sk{a['id']}")
            ncap = st.number_input("Capacity", 1, 100, a["capacity"], key=f"cap{a['id']}")
            if st.button("Save changes", key="save_agent"):
                act("PATCH", f"/agents/{a['id']}", f"{a['name']} updated", roster=True,
                    json={"role": nrole, "skills": nsk, "capacity": int(ncap)})


def page_system():
    ok, s = call("GET", "/stats")
    if not ok:
        st.error(s)
        return
    c = st.columns(5)
    c[0].metric("Tickets", f"{s['total_tickets']:,}")
    c[1].metric("B+ tree height (by id)", s["btree_height"]["by_id"])
    c[2].metric("SLA heap size", f"{s['sla_tracked']:,}")
    c[3].metric("Indexed search words", f"{s['indexed_tokens']:,}")
    c[4].metric("API round trip", f"{st.session_state.get('last_ms', 0):.0f} ms")
    with st.expander("Raw stats"):
        st.json(s)
    if can("system_operations"):
        b1, b2, b3 = st.columns(3)
        if wide_button(b1, "Run dispatcher now", key="run_dispatch"):
            ok, r = call("POST", "/dispatch")
            st.info(f"{len(r)} ticket(s) assigned" if ok else r)
        if wide_button(b2, "Run SLA escalation now", key="run_sla"):
            ok, r = call("POST", "/sla/escalate")
            st.info(f"{len(r)} ticket(s) escalated / flagged" if ok else r)
        if wide_button(b3, "Verify ALL audit chains", key="run_verify"):
            with st.spinner("Streaming the whole audit table…"):
                ok, r = call("GET", "/audit/verify-all")
            (st.success if ok and r["valid"] else st.error)(
                f"{r['entries_checked']:,} entries checked, {r['broken_tickets']} broken chain(s)" if ok else r)
    else:
        st.caption("System operations need a Supervisor or Admin.")
    st.divider()
    st.markdown("**Ticket generator** - create random tickets (one request, one database transaction)")
    n = st.slider("How many tickets", 1, 500, 20)
    if st.button("Generate tickets"):
        subjects = {"billing": "Double charge on invoice", "login": "Cannot log in to account",
                    "outage": "API returns 500 errors", "shipping": "Order delayed in transit",
                    "bug": "Export to CSV crashes", "feature": "Request dark mode"}
        batch = []
        for _ in range(n):
            cat = random.choice(list(subjects))
            batch.append({"subject": subjects[cat], "description": "demo ticket", "category": cat,
                          "customer": f"demo{random.randrange(40)}@example.com",
                          "priority": random.choices([1, 2, 3, 4], weights=[10, 20, 40, 30])[0]})
        ok, r = call("POST", "/tickets/bulk", json={"tickets": batch})
        if ok:
            st.success(f"{r['created']} ticket(s) created in {st.session_state['last_ms']:.0f} ms "
                       f"(#{r['first_id']}–#{r['last_id']}), {r['auto_assigned']} auto-assigned. See the Live board.")
        else:
            st.error(r)


st.title("Customer Support Ticket Router")
if page == PAGES[0]:
    st.fragment(run_every=3 if AUTO else None)(live_board)(LIMIT)
elif page == PAGES[1]:
    page_new()
elif page == PAGES[2]:
    page_search()
elif page == PAGES[3]:
    page_ticket()
elif page == PAGES[4]:
    page_agents()
else:
    page_system()
