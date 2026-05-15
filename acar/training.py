from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import Example
from .models import ACAHyperNetwork, InterventionSpec, LLMBackend, make_prompt_for_example, program_to_ids


@dataclass
class Batch:
    input_ids: torch.Tensor
    prog_ids: torch.Tensor
    abstract_values: torch.Tensor
    labels: torch.Tensor


def _question_from_program(ex: Example) -> str:
    return f"Infer final entity for split={ex.split} with {len(ex.program.steps)} symbolic steps."


def collate_examples(
    examples: Sequence[Example],
    llm: LLMBackend,
    num_entities: int,
    max_prog_len: int,
    d_model: int,
    device: str,
) -> Batch:
    labels = torch.tensor([ex.answer for ex in examples], dtype=torch.long, device=device)
    prog = torch.tensor([program_to_ids(ex.program, max_prog_len) for ex in examples], dtype=torch.long, device=device)

    prompts = [make_prompt_for_example(ex.context, _question_from_program(ex)) for ex in examples]
    input_ids = llm.tokenize_batch(prompts, device=device)

    abstract_values = torch.zeros((len(examples) * max_prog_len, d_model), dtype=torch.float32, device=device)
    for bi, ex in enumerate(examples):
        for si, step in enumerate(ex.program.steps[:max_prog_len]):
            idx = bi * max_prog_len + si
            col = (int(step.primitive) * 131 + ex.answer) % d_model
            abstract_values[idx, col] = 1.0
    return Batch(input_ids=input_ids, prog_ids=prog, abstract_values=abstract_values, labels=labels)


class ACATrainer:
    def __init__(self, llm: LLMBackend, hyper: ACAHyperNetwork, lr: float, weight_decay: float, grad_clip: float, device: str):
        self.llm = llm
        self.hyper = hyper.to(device)
        self.device = device
        self.grad_clip = grad_clip
        self.opt = torch.optim.AdamW(self.hyper.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(self, batch: Batch) -> float:
        self.hyper.train()
        self.opt.zero_grad(set_to_none=True)
        spec = self.hyper(batch.prog_ids)
        logits = self.llm.forward_with_intervention(batch.input_ids, batch.abstract_values, spec)
        # Model logits can be vocab-sized; map label to available index safely.
        y = batch.labels % logits.size(-1)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hyper.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> Tuple[float, float]:
        self.hyper.eval()
        spec = self.hyper(batch.prog_ids)
        logits = self.llm.forward_with_intervention(batch.input_ids, batch.abstract_values, spec)
        y = batch.labels % logits.size(-1)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        return float(loss), float(acc)
