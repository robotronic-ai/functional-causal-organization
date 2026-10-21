#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TARGET-BOUNDARY exact candidate v0.2.

Written after TARGET_BOUNDARY_V02_PRE_CANDIDATE_FREEZE.zip.

Admissible boundary =
  structural causal cut
  AND
  exact interventional sufficiency.
"""

from __future__ import annotations

from itertools import combinations, product


class TargetBoundaryExactV02:
    name="TARGET-BOUNDARY-EXACT"
    version="0.2"

    @staticmethod
    def powerset(items):
        items=tuple(items)
        for r in range(len(items)+1):
            for c in combinations(items,r):
                yield frozenset(c)

    @staticmethod
    def structural_cut(row,B):
        # Reachability from source in the graph mutilated by do(B):
        # incoming arrows into B are removed.
        children={}
        for n in row["nodes"]:
            for p in n["parents"]:
                if n["id"] in B:
                    continue
                children.setdefault(p,[]).append(n["id"])

        seen={row["source"]}
        stack=[row["source"]]
        while stack:
            x=stack.pop()
            for y in children.get(x,()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)

        return all(y not in seen for y in row["future_nodes"])

    @staticmethod
    def eval_model(row,source_value,background_values,interventions):
        env={row["source"]:source_value}
        env.update(background_values)
        for n in row["nodes"]:
            vid=n["id"]
            if vid in interventions:
                env[vid]=interventions[vid]
                continue
            key="|".join(str(env[p]) for p in n["parents"])
            env[vid]=n["table"][key]
        return tuple(env[y] for y in row["future_nodes"])

    @classmethod
    def interventionally_sufficient(cls,row,B):
        alph={row["source"]:tuple(row["source_alphabet"])}
        for x in row["background_inputs"]:
            alph[x["id"]]=tuple(x["alphabet"])
        for n in row["nodes"]:
            alph[n["id"]]=tuple(n["alphabet"])

        bvars=tuple(sorted(B))
        bgvars=tuple(x["id"] for x in row["background_inputs"])

        for bgvals in product(*(alph[x] for x in bgvars)):
            bg=dict(zip(bgvars,bgvals))
            for bvals in product(*(alph[x] for x in bvars)):
                iv=dict(zip(bvars,bvals))
                outs=[
                    cls.eval_model(row,z,bg,iv)
                    for z in alph[row["source"]]
                ]
                if any(o!=outs[0] for o in outs[1:]):
                    return False
        return True

    @classmethod
    def admissible(cls,row,B):
        return cls.structural_cut(row,B) and cls.interventionally_sufficient(row,B)

    @classmethod
    def infer(cls,row):
        sufficient=[]
        for B in cls.powerset(row["eligible_next_nodes"]):
            if cls.admissible(row,B):
                sufficient.append(B)

        minimal=[
            B for B in sufficient
            if not any(C < B for C in sufficient)
        ]
        minimal=sorted(minimal,key=lambda b:(len(b),tuple(sorted(b))))

        if not minimal:
            return {
                "status":"TB-V4",
                "minimal_boundary_count":0,
                "minimal_boundaries":[],
            }

        return {
            "status":"TB-V1" if len(minimal)==1 else "TB-V3",
            "minimal_boundary_count":len(minimal),
            "minimal_boundaries":[sorted(b) for b in minimal],
        }


CANDIDATE=TargetBoundaryExactV02()
