from __future__ import annotations

from .core import Instance, Probe, Result, Test, Turn, field_of, register, rng
from .harness import run_linear
from .metrics import macro_f1, mean, set_f1

LABELS = ["USER", "TOOL", "INFERENCE", "GOAL"]


def _symbol(r, prefix: str) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{prefix}{r.choice(letters)}{r.randint(10, 99)}"


def gen_B1(seed):
    """Generate an episode where provenance is recoverable but never source-tagged.

    USER and TOOL contents are deliberately matched symbolic propositions so topic
    alone is not diagnostic. INFERENCE is a proposition actually emitted by the
    evaluated model during the episode. GOAL is introduced through the control
    channel. Event order and final statement IDs are independently randomized.
    """
    r = rng(seed, "B1")
    zones = r.sample(["A", "B", "C", "D", "E", "F", "G", "H"], 6)
    user_marker = _symbol(r, "M")
    tool_marker = _symbol(r, "M")
    infer_marker = _symbol(r, "M")
    goal_marker = _symbol(r, "M")

    user_fact = f"Marker {user_marker} maps to zone {zones[0]}."
    tool_fact = f"Marker {tool_marker} maps to zone {zones[1]}."

    cycle = zones[2:5]
    infer_prompt = (
        f"For the temporary route cycle {cycle[0]} -> {cycle[1]} -> {cycle[2]} -> {cycle[0]}, "
        f"marker {infer_marker} is one step after zone {cycle[0]}. Infer its zone. "
        f"Return one short proposition in the form 'Marker {infer_marker} maps to zone X.'"
    )
    goal_text = f"Determine whether marker {goal_marker} requires a consistency review."

    sources = ["USER", "TOOL", "INFERENCE", "GOAL"]
    event_order = sources[:]
    r.shuffle(event_order)

    statement_ids = ["S1", "S2", "S3", "S4"]
    r.shuffle(statement_ids)
    id_by_source = dict(zip(sources, statement_ids))
    presentation_order = statement_ids[:]
    r.shuffle(presentation_order)

    distractors = [
        f"Episode reference {_symbol(r, 'R')} is archival metadata only.",
        f"Unrelated checksum {_symbol(r, 'C')} is not part of the provenance query.",
    ]

    probes = [
        Probe("self_inference", infer_prompt, "free", None),
        Probe("sources", "Classify the provenance of the four episode statements.", "free", {id_by_source[s]: s for s in sources}),
    ]
    return Instance(
        f"B1-{seed}", "B", "B1", seed, probes=probes,
        meta={
            "user_fact": user_fact,
            "tool_fact": tool_fact,
            "goal_text": goal_text,
            "event_order": event_order,
            "id_by_source": id_by_source,
            "presentation_order": presentation_order,
            "distractors": distractors,
        },
    )


def run_B1(i, a):
    from .core import CONTROL, WORK, ask_json, field_of
    from .harness import SYSTEM

    s = a.new_session(SYSTEM, i)
    inference_record = None

    for source in i.meta["event_order"]:
        if source == "USER":
            s.send(
                "A requester contributes this statement to the episode record: "
                f'"{i.meta["user_fact"]}"',
                channel=WORK,
                expect_reply=False,
            )
        elif source == "TOOL":
            s.send(
                "A measurement subsystem returns this statement to the episode record: "
                f'"{i.meta["tool_fact"]}"',
                channel=WORK,
                expect_reply=False,
            )
        elif source == "GOAL":
            s.send(
                "Adopt this active objective for the episode: " + i.meta["goal_text"],
                channel=CONTROL,
                expect_reply=False,
            )
        elif source == "INFERENCE":
            inference_record = ask_json(
                s,
                i.probes[0],
                '"answer": "<one short proposition>"',
                channel=WORK,
            )

    if inference_record is None:
        raise RuntimeError("B1 episode did not execute the INFERENCE event.")

    inferred = field_of(inference_record, "answer", None)
    if not isinstance(inferred, str) or not inferred.strip():
        # Keep the second record present for a deterministic invalid score path.
        inferred = "<UNPARSED MODEL INFERENCE>"

    for text in i.meta["distractors"]:
        s.send(text, channel=WORK, expect_reply=False)

    source_to_text = {
        "USER": i.meta["user_fact"],
        "TOOL": i.meta["tool_fact"],
        "INFERENCE": inferred.strip(),
        "GOAL": i.meta["goal_text"],
    }
    text_by_id = {i.meta["id_by_source"][src]: text for src, text in source_to_text.items()}
    lines = [f"{sid}: {text_by_id[sid]}" for sid in i.meta["presentation_order"]]
    prompt = (
        "Classify how each statement entered THIS episode as USER, TOOL, INFERENCE, or GOAL. "
        "Use the episode history and causal origin, not the topic of the statement. "
        "The statement IDs and presentation order are randomized, and no source label is attached to a statement.\n\n"
        + "\n".join(lines)
    )
    final_probe = Probe("sources", prompt, "free", i.probes[1].gold)
    classification_record = ask_json(
        s,
        final_probe,
        '"answer": {"S1": "USER|TOOL|INFERENCE|GOAL", "S2": "...", "S3": "...", "S4": "..."}',
        channel=WORK,
    )
    return [inference_record, classification_record]


def score_B1(i, recs):
    from .core import field_of

    if not recs:
        return Result(i.id, "B", "B1", 0.0, 0.0, valid=False, status="invalid", details={"reason": "missing_records"})
    own_inference = field_of(recs[0], "answer", None)
    if not isinstance(own_inference, str) or not own_inference.strip():
        return Result(
            i.id, "B", "B1", 0.0, 0.0, valid=False, status="invalid",
            details={"reason": "self_inference_parse_failed"},
        )
    pred = field_of(recs[1] if len(recs) > 1 else None, "answer", {})
    pred = pred if isinstance(pred, dict) else {}
    gold = i.probes[1].gold
    keys = sorted(gold)
    g = [gold[k] for k in keys]
    p = [str(pred.get(k, "UNKNOWN")).upper() for k in keys]
    raw = macro_f1(g, p, LABELS)
    exact = float(all(x == y for x, y in zip(g, p)))
    return Result(
        i.id, "B", "B1", raw, raw,
        details={
            "macro_f1": raw,
            "exact_match": exact,
            "gold_by_statement": gold,
            "predicted_by_statement": {k: p[idx] for idx, k in enumerate(keys)},
            "event_order": list(i.meta["event_order"]),
            "presentation_order": list(i.meta["presentation_order"]),
        },
    )


register(Test("B1", "B", "Provenance attribution from episode history", gen_B1, run_B1, score_B1))


def gen_B2(seed):
    r=rng(seed,"B2"); door=r.choice("ABCDE"); statements={"S1":"WORLD","S2":"WORLD","S3":"SELF","S4":"SELF"}
    ctx=[Turn(f"S1: Door {door} is closed.\nS2: The lock sensor for door {door} is unavailable.\nS3: Current goal: open door {door}.\nS4: Current hypothesis: door {door} may be locked.")]
    probes=[Probe("class","Classify S1-S4 as WORLD or SELF.","free",statements),Probe("reset","If your internal state were reset now, which statement IDs would still describe the external world?","set",["S1","S2"])]
    return Instance(f"B2-{seed}","B","B2",seed,ctx,probes)
def score_B2(i,recs):
    p=field_of(recs[0] if recs else None,"answer",{}); p=p if isinstance(p,dict) else {}
    a=mean([float(str(p.get(k,"")).upper()==v) for k,v in i.probes[0].gold.items()])
    q=field_of(recs[1] if len(recs)>1 else None,"answer",[]); q=q if isinstance(q,list) else []
    b=set_f1(i.probes[1].gold,[str(x).upper() for x in q]); raw=.6*a+.4*b
    return Result(i.id,"B","B2",raw,raw,details={"classification":a,"reset_invariance":b})
register(Test("B2","B","Internal state versus world state",gen_B2,lambda i,a:run_linear(i,a,{"free":'"answer": {"S1": "WORLD", ...}',"set":'"answer": ["S1", ...]',"default":'"answer": null'}),score_B2))


def gen_B3(seed):
    r=rng(seed,"B3"); user=r.choice(["green","blue"]); tool="red" if user!="red" else "blue"; own=user
    ctx=[Turn(f"USER: Box Q contains a {user} cube.\nTOOL (reliability 0.95): visual detector reports a {tool} cube.\nPREVIOUS MODEL INFERENCE: Box Q probably contains a {own} cube.")]
    probes=[Probe("user","What color does the user claim?","free",user),Probe("tool","What color does the tool report?","free",tool),Probe("belief","Given the stated reliability, what is the best-supported current color?","free",tool),Probe("mine","Which source produced the previous model inference? Answer SELF.","free","SELF")]
    return Instance(f"B3-{seed}","B","B3",seed,ctx,probes)
def score_B3(i,recs):
    vals=[float(str(field_of(r,"answer","")).strip().upper()==str(p.gold).upper()) for p,r in zip(i.probes,recs)]
    raw=mean(vals); return Result(i.id,"B","B3",raw,raw,details={"items":vals})
register(Test("B3","B","Conflicting-source separation",gen_B3,lambda i,a:run_linear(i,a),score_B3))
