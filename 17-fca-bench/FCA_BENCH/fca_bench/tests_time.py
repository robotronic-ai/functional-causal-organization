from __future__ import annotations

from datetime import datetime, timedelta
import math
import re

from .core import CONTROL, Instance, Probe, Record, Result, Test, Turn, ask_json, field_of, register, rng
from .harness import SYSTEM, push_context
from .metrics import mean, set_f1


def _hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def gen_T1(seed: int) -> Instance:
    r = rng(seed, "T1")
    names = [f"P{c}" for c in "ABCDE"]
    events = []
    state = {n: "NOT_STARTED" for n in names}
    now = datetime(2026, 1, 1, 9, 0)
    eid = 0
    for _ in range(r.randint(7, 11)):
        now += timedelta(minutes=r.randint(1, 9))
        name = r.choice(names)
        allowed = {
            "NOT_STARTED": ["START"],
            "ACTIVE": ["PAUSE", "CANCEL", "COMPLETE"],
            "PAUSED": ["RESUME", "CANCEL"],
            "CANCELLED": [],
            "COMPLETE": [],
        }[state[name]]
        if not allowed:
            continue
        action = r.choice(allowed)
        state[name] = {"START":"ACTIVE","PAUSE":"PAUSED","RESUME":"ACTIVE","CANCEL":"CANCELLED","COMPLETE":"COMPLETE"}[action]
        eid += 1
        events.append((f"E{eid}", now, action, name))
    shuffled = list(events)
    r.shuffle(shuffled)
    lines = [f"{eid} [{_hhmm(t)}] {action} {name}" for eid, t, action, name in shuffled]
    context = [Turn("Project event log, deliberately shuffled:\n" + "\n".join(lines))]
    probes = [
        Probe("order", "Return the event IDs in true chronological order.", "set", [e[0] for e in events]),
        Probe("state", "Return the current state of every project as an object project -> state. Use NOT_STARTED, ACTIVE, PAUSED, CANCELLED, or COMPLETE.", "free", state),
    ]
    return Instance(f"T1-{seed}", "T", "T1", seed, context, probes)


def run_T1(inst, adapter):
    s = adapter.new_session(SYSTEM, inst); push_context(s, inst)
    a = ask_json(s, inst.probes[0], '"answer": ["E1", "E2", ...]')
    b = ask_json(s, inst.probes[1], '"answer": {"PA": "ACTIVE", ...}')
    return [a,b]


def score_T1(inst, recs):
    order = field_of(recs[0] if recs else None, "answer", [])
    if not isinstance(order, list): order = []
    gold = inst.probes[0].gold
    pos = {x:i for i,x in enumerate(order)}
    pairs = [(gold[i], gold[j]) for i in range(len(gold)) for j in range(i+1,len(gold))]
    order_score = mean([float(a in pos and b in pos and pos[a] < pos[b]) for a,b in pairs]) if pairs else 1.0
    pred_state = field_of(recs[1] if len(recs)>1 else None, "answer", {})
    if not isinstance(pred_state, dict): pred_state = {}
    state_score = mean([float(str(pred_state.get(k,"")).upper()==v) for k,v in inst.probes[1].gold.items()])
    raw = 0.65*order_score+0.35*state_score
    return Result(inst.id,"T","T1",raw,raw,details={"order":order_score,"state":state_score})

register(Test("T1","T","Event timeline reconstruction",gen_T1,run_T1,score_T1))



def _interval_rows(prefix: str, times: list[datetime], r):
    labels = [f"{prefix}{x}" for x in r.sample(range(1, 10), len(times) - 1)]
    intervals = []
    for idx, label in enumerate(labels):
        intervals.append({
            "id": label,
            "start": times[idx],
            "end": times[idx + 1],
            "minutes": int((times[idx + 1] - times[idx]).total_seconds() // 60),
            "active": idx % 2 == 0,
        })
    shown = list(intervals)
    r.shuffle(shown)
    lines = [f"{x['id']} = [{_hhmm(x['start'])}, {_hhmm(x['end'])})" for x in shown]
    return intervals, lines


def gen_T2(seed:int)->Instance:
    """Pause-aware temporal accounting with arithmetic separated from the primary score.

    The primary FCA score asks which intervals causally consume active time. Numeric
    duration answers are still collected, but only as an arithmetic diagnostic.
    This prevents general arithmetic competence from dominating Temporal Grounding.
    """
    r = rng(seed, "T2")

    # Task K: same semantic structure as earlier T2, but interval IDs and their
    # presentation order are randomized so there is no fixed K1/K3 shortcut.
    start = datetime(2026, 1, 1, 14, 0) + timedelta(minutes=r.randint(0, 30))
    a = r.randint(4, 12)
    pause = r.randint(5, 15)
    b = r.randint(6, 18)
    t1 = start + timedelta(minutes=a)
    t2 = t1 + timedelta(minutes=pause)
    t3 = t2 + timedelta(minutes=b)
    k_intervals, k_lines = _interval_rows("K", [start, t1, t2, t3], r)

    # Deadline E: the countdown consumes only active (non-paused) intervals.
    deadline = r.randint(30, 70)
    stop = r.randint(5, 15)
    resume_gap = r.randint(5, 15)
    after = r.randint(5, 25)
    d0 = datetime(2026, 1, 1, 10, 0)
    d1 = d0 + timedelta(minutes=stop)
    d2 = d1 + timedelta(minutes=resume_gap)
    d3 = d2 + timedelta(minutes=after)
    e_intervals, e_lines = _interval_rows("E", [d0, d1, d2, d3], r)

    k_active = [x["id"] for x in k_intervals if x["active"]]
    e_active = [x["id"] for x in e_intervals if x["active"]]
    active_minutes = sum(x["minutes"] for x in k_intervals if x["active"])
    elapsed_deadline_active = sum(x["minutes"] for x in e_intervals if x["active"])
    remaining = max(0, deadline - elapsed_deadline_active)

    context = [
        Turn(
            f"[{_hhmm(start)}] START task K\n"
            f"[{_hhmm(t1)}] PAUSE task K\n"
            f"[{_hhmm(t2)}] RESUME task K\n"
            f"[{_hhmm(t3)}] STOP task K\n\n"
            "Candidate intervals for task K (identifiers are arbitrary):\n"
            + "\n".join(k_lines)
        ),
        Turn(
            f"[{_hhmm(d0)}] Create deadline E for {deadline} active minutes\n"
            f"[{_hhmm(d1)}] PAUSE deadline E\n"
            f"[{_hhmm(d2)}] RESUME deadline E\n"
            f"[{_hhmm(d3)}] Current time for deadline query\n\n"
            "Candidate intervals for deadline E (identifiers are arbitrary):\n"
            + "\n".join(e_lines)
        ),
    ]

    probes = [
        Probe(
            "active_intervals",
            "Which interval IDs contribute to task K's actual active time? "
            "Select intervals from the candidate list; paused time must not count.",
            "set",
            k_active,
        ),
        Probe(
            "deadline_intervals",
            "Which interval IDs have consumed deadline E's active-minute budget before the current time? "
            "Select intervals from the candidate list; paused time must not count.",
            "set",
            e_active,
        ),
        # Numeric probes are diagnostics only and do not contribute to the FCA T2 score.
        Probe("active_numeric", "How many minutes was task K actually active?", "num", active_minutes),
        Probe("remaining_numeric", "How many active minutes remain on deadline E?", "num", remaining),
    ]

    meta = {
        "k_intervals": [
            {"id": x["id"], "minutes": x["minutes"], "active": x["active"]}
            for x in k_intervals
        ],
        "e_intervals": [
            {"id": x["id"], "minutes": x["minutes"], "active": x["active"]}
            for x in e_intervals
        ],
        "deadline_minutes": deadline,
        "active_minutes_gold": active_minutes,
        "remaining_minutes_gold": remaining,
    }
    return Instance(f"T2-{seed}", "T", "T2", seed, context, probes, meta=meta)


def run_T2(i,a):
    records = []
    for p in i.probes:
        # Fresh session per probe prevents the previous answer from becoming a
        # causal cue for the next probe while preserving the identical episode.
        s = a.new_session(SYSTEM, i)
        push_context(s, i)
        schema = '"answer": ["<interval_id>", ...]' if p.kind == "set" else '"answer": <integer>'
        records.append(ask_json(s, p, schema))
    return records


_NUMERIC_TOKEN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def _normalized_numeric(value):
    if isinstance(value, bool) or value is None:
        return None, False
    if isinstance(value, (int, float)):
        number = float(value)
        return (number, True) if math.isfinite(number) else (None, True)
    if isinstance(value, str):
        tokens = _NUMERIC_TOKEN.findall(value.strip())
        if len(tokens) != 1:
            return None, False
        try:
            number = float(tokens[0])
        except ValueError:
            return None, False
        return (number, False) if math.isfinite(number) else (None, False)
    return None, False


def _answer_list(record):
    value = field_of(record, "answer", [])
    if not isinstance(value, list):
        return []
    return [str(x).strip().upper() for x in value if str(x).strip()]


def _balanced_accuracy(universe, gold, pred):
    """Balanced accuracy of the membership classification over `universe`."""
    g, p, u = set(gold), set(pred), set(universe)
    pos, neg = u & g, u - g
    tpr = len(pos & p) / len(pos) if pos else 1.0
    tnr = len(neg - p) / len(neg) if neg else 1.0
    return 0.5 * (tpr + tnr)


def score_T2(i,recs):
    # Primary temporal score: interval accounting only, no arithmetic required.
    active_pred = _answer_list(recs[0] if len(recs) > 0 else None)
    deadline_pred = _answer_list(recs[1] if len(recs) > 1 else None)
    active_gold = [str(x).upper() for x in i.probes[0].gold]
    deadline_gold = [str(x).upper() for x in i.probes[1].gold]

    # Interval membership: every interval in the probe universe is classified
    # as counting (ACTIVE) or not counting (PAUSED); the score is the balanced
    # accuracy of that classification.
    k_ids = [x["id"].upper() for x in i.meta["k_intervals"]]
    e_ids = [x["id"].upper() for x in i.meta["e_intervals"]]
    active_ba = _balanced_accuracy(k_ids, active_gold, active_pred)
    deadline_ba = _balanced_accuracy(e_ids, deadline_gold, deadline_pred)
    temporal_score = mean([active_ba, deadline_ba])
    active_selection = set_f1(active_gold, active_pred)
    deadline_selection = set_f1(deadline_gold, deadline_pred)

    # Secondary arithmetic diagnostic. It is deliberately excluded from raw/score.
    numeric_details = []
    numeric_correct = []
    for probe_index in (2, 3):
        p = i.probes[probe_index]
        r = recs[probe_index] if len(recs) > probe_index else None
        answer = field_of(r, "answer", None)
        normalized, native_numeric = _normalized_numeric(answer)
        correct = float(normalized is not None and abs(normalized - float(p.gold)) < 1e-9)
        numeric_correct.append(correct)
        numeric_details.append({
            "probe_id": p.id,
            "gold": p.gold,
            "parsed_answer": answer,
            "normalized_numeric": normalized,
            "native_json_number": native_numeric,
            "correct": bool(correct),
            "raw_response": r.raw if r is not None else "",
        })

    # Derived numeric consequences of the model's interval selections, calculated
    # by the harness rather than by the model.
    k_minutes = {x["id"].upper(): x["minutes"] for x in i.meta["k_intervals"]}
    e_minutes = {x["id"].upper(): x["minutes"] for x in i.meta["e_intervals"]}
    derived_active = sum(k_minutes.get(x, 0) for x in set(active_pred))
    derived_consumed = sum(e_minutes.get(x, 0) for x in set(deadline_pred))
    derived_remaining = max(0, int(i.meta["deadline_minutes"]) - derived_consumed)

    return Result(
        i.id, "T", "T2", temporal_score, temporal_score,
        details={
            "primary_metric": "test_level_chance_corrected_balanced_accuracy_of_pause_aware_interval_membership",
            "raw_balanced_accuracy": temporal_score,
            "aggregate_transform": "max(0, 2 * mean(raw_balanced_accuracy) - 1)",
            "active_interval_balanced_accuracy": active_ba,
            "deadline_interval_balanced_accuracy": deadline_ba,
            "active_interval_set_f1_diagnostic": active_selection,
            "deadline_interval_set_f1_diagnostic": deadline_selection,
            "active_exact_set": float(set(active_gold) == set(active_pred)),
            "deadline_exact_set": float(set(deadline_gold) == set(deadline_pred)),
            "active_universe": k_ids,
            "deadline_universe": e_ids,
            "active_gold": active_gold,
            "active_predicted": active_pred,
            "deadline_gold": deadline_gold,
            "deadline_predicted": deadline_pred,
            "derived_active_minutes_from_selection": derived_active,
            "derived_remaining_minutes_from_selection": derived_remaining,
            "gold_active_minutes": i.meta["active_minutes_gold"],
            "gold_remaining_minutes": i.meta["remaining_minutes_gold"],
            "numeric_diagnostic_accuracy": mean(numeric_correct),
            "numeric_diagnostics": numeric_details,
            "note": "Numeric arithmetic probes are diagnostic and excluded from the FCA T2 score. The stored instance score is the raw balanced accuracy; the T2 aggregate is chance-corrected after averaging instances.",
        },
    )


register(Test(
    "T2",
    "T",
    "Pause-aware temporal accounting (chance-corrected, arithmetic-controlled)",
    gen_T2,
    run_T2,
    score_T2,
    notes=(
        "The model selects which randomized intervals causally count as active; "
        "the harness computes their duration. The score is the test-level "
        "chance-corrected balanced accuracy of interval membership, "
        "max(0, 2*mean(BA) - 1). Direct numeric answers are retained only as an "
        "arithmetic diagnostic."
    ),
))

def gen_T3(seed:int)->Instance:
    r=rng(seed,"T3"); now=datetime(2026,1,1,8,0)+timedelta(minutes=r.randint(0,300)); goals=[]
    ctx=[]
    for idx in range(3):
        now+=timedelta(minutes=r.randint(3,20)); goal=f"G{idx+1}-{r.choice(['COST','LATENCY','TESTS','MEMORY'])}"; goals.append((goal,now)); ctx.append(Turn(f"CONTROL_TIMESTAMP={_hhmm(now)}; ACTIVE_GOAL={goal}",channel=CONTROL))
    now+=timedelta(minutes=r.randint(3,20)); ctx.append(Turn(f"CONTROL_TIMESTAMP={_hhmm(now)}",channel=CONTROL))
    goal,tg=goals[-1]
    probes=[Probe("goal","What is the currently active goal?","free",goal),Probe("activated",f"At what time was goal {goal} activated? Return HH:MM.","free",_hhmm(tg)),Probe("age",f"How many minutes ago was goal {goal} activated?","num",int((now-tg).total_seconds()/60))]
    return Instance(f"T3-{seed}","T","T3",seed,ctx,probes)

def run_T3(i,a):
    s=a.new_session(SYSTEM,i); push_context(s,i); return [ask_json(s,p,'"answer": <value>') for p in i.probes]

def score_T3(i,recs):
    vals=[]
    for p,r in zip(i.probes,recs):
        got=field_of(r,"answer")
        if p.kind=="num":
            try: vals.append(float(float(got)==p.gold))
            except Exception: vals.append(0.0)
        else: vals.append(float(str(got).strip().upper()==str(p.gold).upper()))
    raw=mean(vals); return Result(i.id,"T","T3",raw,raw,details={"items":vals})
register(Test("T3","T","Temporal self-location",gen_T3,run_T3,score_T3,notes="Run a matched WORK-channel condition to separate channel effects from architecture effects."))
