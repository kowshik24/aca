from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Sequence, Set, Tuple

Triple = Tuple[int, int, int]
EntitySet = Set[int]


class Primitive(IntEnum):
    LOAD_CONTEXT = 0
    QUERY_1HOP = 1
    EXPAND_SET = 2
    FILTER = 3
    ARGMAX = 4
    ARGMIN = 5
    INTERSECT = 6
    OUTPUT = 7


@dataclass
class ProgramStep:
    primitive: Primitive
    args: Tuple


@dataclass
class Program:
    steps: List[ProgramStep]


class RelationalDSL:
    def __init__(self, num_entities: int):
        self.num_entities = num_entities

    @staticmethod
    def load_context(triples: Sequence[Triple]) -> List[Triple]:
        return list(triples)

    @staticmethod
    def query_1hop(context: Sequence[Triple], entity: int, relation: int, direction: str = "out") -> EntitySet:
        if direction == "out":
            return {dst for src, rel, dst in context if src == entity and rel == relation}
        return {src for src, rel, dst in context if dst == entity and rel == relation}

    def expand_set(self, context: Sequence[Triple], entities: EntitySet, relation: int, direction: str = "out") -> EntitySet:
        out: EntitySet = set()
        for ent in entities:
            out |= self.query_1hop(context, ent, relation, direction)
        return out

    @staticmethod
    def filter_set(
        context: Sequence[Triple],
        entities: EntitySet,
        relation: int,
        direction: str,
        target: int,
    ) -> EntitySet:
        out: EntitySet = set()
        for ent in entities:
            neigh = ({dst for src, rel, dst in context if src == ent and rel == relation} if direction == "out"
                     else {src for src, rel, dst in context if dst == ent and rel == relation})
            if target in neigh:
                out.add(ent)
        return out

    @staticmethod
    def argmax_set(context: Sequence[Triple], entities: EntitySet, relation: int, direction: str = "out") -> int:
        scored = []
        for ent in entities:
            neigh = ({dst for src, rel, dst in context if src == ent and rel == relation} if direction == "out"
                     else {src for src, rel, dst in context if dst == ent and rel == relation})
            scored.append((len(neigh), ent))
        return max(scored)[1] if scored else -1

    @staticmethod
    def argmin_set(context: Sequence[Triple], entities: EntitySet, relation: int, direction: str = "out") -> int:
        scored = []
        for ent in entities:
            neigh = ({dst for src, rel, dst in context if src == ent and rel == relation} if direction == "out"
                     else {src for src, rel, dst in context if dst == ent and rel == relation})
            scored.append((len(neigh), ent))
        return min(scored)[1] if scored else -1

    @staticmethod
    def intersect(a: EntitySet, b: EntitySet) -> EntitySet:
        return a & b

    @staticmethod
    def output_entity(ent: int) -> int:
        return ent

    def execute(self, program: Program, context: Sequence[Triple]) -> Tuple[int, List[object]]:
        mem: Dict[str, object] = {}
        trace: List[object] = []
        for idx, step in enumerate(program.steps):
            p = step.primitive
            args = step.args
            if p == Primitive.LOAD_CONTEXT:
                mem[f"s{idx}"] = self.load_context(context)
            elif p == Primitive.QUERY_1HOP:
                ent, rel, d = args
                mem[f"s{idx}"] = self.query_1hop(context, ent, rel, d)
            elif p == Primitive.EXPAND_SET:
                src_key, rel, d = args
                mem[f"s{idx}"] = self.expand_set(context, mem[src_key], rel, d)
            elif p == Primitive.FILTER:
                src_key, rel, d, target = args
                mem[f"s{idx}"] = self.filter_set(context, mem[src_key], rel, d, target)
            elif p == Primitive.ARGMAX:
                src_key, rel, d = args
                mem[f"s{idx}"] = self.argmax_set(context, mem[src_key], rel, d)
            elif p == Primitive.ARGMIN:
                src_key, rel, d = args
                mem[f"s{idx}"] = self.argmin_set(context, mem[src_key], rel, d)
            elif p == Primitive.INTERSECT:
                a_key, b_key = args
                mem[f"s{idx}"] = self.intersect(mem[a_key], mem[b_key])
            elif p == Primitive.OUTPUT:
                src_key = args[0]
                mem[f"s{idx}"] = self.output_entity(mem[src_key])
            trace.append(mem[f"s{idx}"])
        return int(trace[-1]), trace
