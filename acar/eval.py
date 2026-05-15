from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .data import Example
from .dsl import Primitive
from .models import ACAHyperNetwork, LLMBackend, program_to_ids
from .training import collate_examples


@dataclass
class RewriteRule:
    primitive: Primitive
    layer_read_mode: int
    layer_write_mode: int
    position_write_mode: int


def extract_rules(model: ACAHyperNetwork, examples: Sequence[Example], max_prog_len: int, device: str) -> List[RewriteRule]:
    model.eval()
    buckets = defaultdict(list)
    with torch.no_grad():
        for ex in examples:
            prog = torch.tensor([program_to_ids(ex.program, max_prog_len)], dtype=torch.long, device=device)
            spec = model(prog)
            for i, step in enumerate(ex.program.steps[:max_prog_len]):
                buckets[step.primitive].append((
                    int(spec.layer_read[i].item()),
                    int(spec.layer_write[i].item()),
                    int(spec.position_write[i].item()),
                ))

    rules: List[RewriteRule] = []
    for primitive, vals in buckets.items():
        (lr, lw, pw), _ = Counter(vals).most_common(1)[0]
        rules.append(RewriteRule(primitive=primitive, layer_read_mode=lr, layer_write_mode=lw, position_write_mode=pw))
    return sorted(rules, key=lambda r: int(r.primitive))


@torch.no_grad()
def counterfactual_faithfulness(
    llm: LLMBackend,
    hyper: ACAHyperNetwork,
    examples: Sequence[Example],
    num_entities: int,
    max_prog_len: int,
    d_model: int,
    corruption_scale: float,
    device: str,
) -> float:
    total = 0
    success = 0
    for ex in examples:
        b = collate_examples([ex], llm=llm, num_entities=num_entities, max_prog_len=max_prog_len, d_model=d_model, device=device)
        spec = hyper(b.prog_ids)
        logits_ok = llm.forward_with_intervention(b.input_ids, b.abstract_values, spec)
        pred_ok = logits_ok.argmax(dim=-1)

        vals_bad = b.abstract_values.clone().mul_(corruption_scale).neg_()
        logits_bad = llm.forward_with_intervention(b.input_ids, vals_bad, spec)
        pred_bad = logits_bad.argmax(dim=-1)

        target = b.labels % logits_ok.size(-1)
        good = bool(pred_ok.item() == target.item())
        changed = bool(pred_bad.item() != pred_ok.item())
        success += int(good and changed)
        total += 1
    return success / max(total, 1)


@torch.no_grad()
def estimate_emi(llm_logits: torch.Tensor, intervened_logits: torch.Tensor) -> float:
    p = F.softmax(llm_logits, dim=-1)
    q = F.softmax(intervened_logits, dim=-1)
    m = 0.5 * (p + q)
    kl1 = torch.sum(p * (p.clamp_min(1e-9).log() - m.clamp_min(1e-9).log()), dim=-1)
    kl2 = torch.sum(q * (q.clamp_min(1e-9).log() - m.clamp_min(1e-9).log()), dim=-1)
    js = 0.5 * (kl1 + kl2)
    return float((1.0 - js.mean()).item())


@torch.no_grad()
def evaluate_split(
    llm: LLMBackend,
    hyper: ACAHyperNetwork,
    examples: Sequence[Example],
    num_entities: int,
    max_prog_len: int,
    d_model: int,
    corruption_scale: float,
    device: str,
) -> dict[str, float]:
    if not examples:
        return {"acc": 0.0, "faithfulness": 0.0, "emi": 0.0}

    batch = collate_examples(examples, llm=llm, num_entities=num_entities, max_prog_len=max_prog_len, d_model=d_model, device=device)
    spec = hyper(batch.prog_ids)
    logits_i = llm.forward_with_intervention(batch.input_ids, batch.abstract_values, spec)

    target = batch.labels % logits_i.size(-1)
    acc = float((logits_i.argmax(dim=-1) == target).float().mean().item())

    # baseline forward with zero intervention values for EMI reference.
    logits_base = llm.forward_with_intervention(batch.input_ids, torch.zeros_like(batch.abstract_values), spec)
    emi = estimate_emi(logits_base, logits_i)

    faith = counterfactual_faithfulness(
        llm=llm,
        hyper=hyper,
        examples=examples,
        num_entities=num_entities,
        max_prog_len=max_prog_len,
        d_model=d_model,
        corruption_scale=corruption_scale,
        device=device,
    )
    return {"acc": acc, "faithfulness": faith, "emi": emi}
