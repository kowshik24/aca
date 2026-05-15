from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence, Tuple

from .dsl import Primitive, Program, ProgramStep, RelationalDSL, Triple

SplitName = Literal["train", "iid", "length", "relation", "novel_comp"]


@dataclass
class Example:
    context: List[Triple]
    program: Program
    answer: int
    level: int
    split: SplitName
    relation_vocab: Tuple[int, ...]


class MultiHopGenerator:
    """Builds blueprint-aligned splits and symbolic traces for ACAR."""

    def __init__(
        self,
        num_entities: int,
        relation_names: Sequence[str],
        test_relation_names: Sequence[str],
        min_edges: int,
        max_edges: int,
    ):
        self.num_entities = num_entities
        all_rel_names = list(relation_names) + list(test_relation_names)
        self.rel2id = {name: i for i, name in enumerate(all_rel_names)}
        self.train_rel_ids = tuple(self.rel2id[r] for r in relation_names)
        self.test_only_rel_ids = tuple(self.rel2id[r] for r in test_relation_names)
        self.all_rel_ids = tuple(self.rel2id.values())
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.dsl = RelationalDSL(num_entities=num_entities)

    def random_graph(self, relation_ids: Sequence[int]) -> List[Triple]:
        n = random.randint(self.min_edges, self.max_edges)
        triples: List[Triple] = []
        seen = set()
        while len(triples) < n:
            t = (
                random.randrange(self.num_entities),
                random.choice(relation_ids),
                random.randrange(self.num_entities),
            )
            if t not in seen:
                seen.add(t)
                triples.append(t)
        return triples

    def _build_chain(self, hops: int, relation_ids: Sequence[int], force_argmax: bool | None = None) -> Program:
        start = random.randrange(self.num_entities)
        rel = random.choice(relation_ids)
        steps = [
            ProgramStep(Primitive.LOAD_CONTEXT, tuple()),
            ProgramStep(Primitive.QUERY_1HOP, (start, rel, "out")),
        ]
        cur = "s1"
        for _ in range(hops - 1):
            steps.append(ProgramStep(Primitive.EXPAND_SET, (cur, rel, "out")))
            cur = f"s{len(steps)-1}"

        argmax = random.random() < 0.5 if force_argmax is None else force_argmax
        if argmax:
            steps.append(ProgramStep(Primitive.ARGMAX, (cur, rel, "out")))
        else:
            steps.append(ProgramStep(Primitive.ARGMIN, (cur, rel, "out")))

        cur = f"s{len(steps)-1}"
        steps.append(ProgramStep(Primitive.OUTPUT, (cur,)))
        return Program(steps=steps)

    def _build_novel_composition(self, relation_ids: Sequence[int]) -> Program:
        # Novel composition: FILTER + ARGMAX path jointly, held-out from train by default.
        start = random.randrange(self.num_entities)
        rel = random.choice(relation_ids)
        target = random.randrange(self.num_entities)
        steps = [
            ProgramStep(Primitive.LOAD_CONTEXT, tuple()),
            ProgramStep(Primitive.QUERY_1HOP, (start, rel, "out")),
            ProgramStep(Primitive.EXPAND_SET, ("s1", rel, "out")),
            ProgramStep(Primitive.FILTER, ("s2", rel, "out", target)),
            ProgramStep(Primitive.ARGMAX, ("s3", rel, "out")),
            ProgramStep(Primitive.OUTPUT, ("s4",)),
        ]
        return Program(steps=steps)

    @staticmethod
    def _level_from_hops(hops: int) -> int:
        if hops == 1:
            return 1
        if hops <= 3:
            return 2
        if hops <= 5:
            return 4
        return 5

    def _make_valid_example(self, split: SplitName, relation_ids: Sequence[int], builder) -> Example:
        program = builder()
        for _ in range(40):
            ctx = self.random_graph(relation_ids)
            answer, _ = self.dsl.execute(program, ctx)
            if 0 <= answer < self.num_entities:
                hops = sum(1 for s in program.steps if s.primitive == Primitive.EXPAND_SET) + 1
                return Example(
                    context=ctx,
                    program=program,
                    answer=answer,
                    level=self._level_from_hops(hops),
                    split=split,
                    relation_vocab=tuple(relation_ids),
                )
        # Final fallback with normalized answer.
        answer = max(answer, 0) % self.num_entities
        return Example(
            context=ctx,
            program=program,
            answer=answer,
            level=self._level_from_hops(1),
            split=split,
            relation_vocab=tuple(relation_ids),
        )

    def build_split(self, split: SplitName, n: int, max_hops: int | None = None) -> List[Example]:
        out: List[Example] = []
        if split in ("train", "iid"):
            rels = self.train_rel_ids
            upper = max_hops or 3
            for _ in range(n):
                hops = random.randint(1, upper)
                # prevent FILTER+ARGMAX during training for novel composition holdout.
                out.append(self._make_valid_example(split, rels, lambda h=hops: self._build_chain(h, rels)))
        elif split == "length":
            rels = self.train_rel_ids
            for _ in range(n):
                hops = random.randint(4, max_hops or 6)
                out.append(self._make_valid_example(split, rels, lambda h=hops: self._build_chain(h, rels)))
        elif split == "relation":
            rels = self.test_only_rel_ids
            for _ in range(n):
                hops = random.randint(1, max_hops or 6)
                out.append(self._make_valid_example(split, rels, lambda h=hops: self._build_chain(h, rels)))
        elif split == "novel_comp":
            rels = self.train_rel_ids
            for _ in range(n):
                out.append(self._make_valid_example(split, rels, lambda: self._build_novel_composition(rels)))
        else:
            raise ValueError(f"Unknown split: {split}")
        return out

    def build_all_splits(
        self,
        train_size: int,
        iid_eval_size: int,
        length_eval_size: int,
        relation_eval_size: int,
        novel_comp_eval_size: int,
        max_hops_train: int,
        max_hops_eval: int,
    ) -> dict[str, List[Example]]:
        return {
            "train": self.build_split("train", train_size, max_hops=max_hops_train),
            "iid": self.build_split("iid", iid_eval_size, max_hops=max_hops_train),
            "length": self.build_split("length", length_eval_size, max_hops=max_hops_eval),
            "relation": self.build_split("relation", relation_eval_size, max_hops=max_hops_eval),
            "novel_comp": self.build_split("novel_comp", novel_comp_eval_size, max_hops=max_hops_eval),
        }
