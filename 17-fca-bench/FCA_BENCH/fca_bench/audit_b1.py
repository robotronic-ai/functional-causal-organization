from __future__ import annotations

from .tests_boundary import gen_B1


def main() -> int:
    seen_id_maps = set()
    seen_event_orders = set()
    for seed in range(20):
        inst = gen_B1(seed)
        gold = inst.probes[1].gold
        if set(gold.values()) != {"USER", "TOOL", "INFERENCE", "GOAL"}:
            raise AssertionError(f"Seed {seed}: invalid B1 gold labels")
        seen_id_maps.add(tuple(sorted((k, v) for k, v in gold.items())))
        seen_event_orders.add(tuple(inst.meta["event_order"]))
        # The scored propositions themselves must not carry bracketed gold source tags.
        for text in (inst.meta["user_fact"], inst.meta["tool_fact"], inst.meta["goal_text"]):
            for label in ("[USER]", "[TOOL]", "[INFERENCE]", "[GOAL]"):
                if label in text:
                    raise AssertionError(f"Seed {seed}: explicit source tag leaked into scored content")
    if len(seen_id_maps) < 4:
        raise AssertionError("B1 statement-ID mapping is insufficiently randomized across seeds")
    if len(seen_event_orders) < 4:
        raise AssertionError("B1 event order is insufficiently randomized across seeds")
    print("B1 design audit PASSED")
    print(f"Distinct ID mappings across 20 seeds: {len(seen_id_maps)}")
    print(f"Distinct event orders across 20 seeds: {len(seen_event_orders)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
