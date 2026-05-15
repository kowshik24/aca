from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dsl import Primitive, Program


@dataclass
class InterventionSpec:
    layer_read: torch.Tensor
    layer_write: torch.Tensor
    position_write: torch.Tensor
    gain: torch.Tensor
    bias: torch.Tensor


class LLMBackend(Protocol):
    d_model: int
    num_layers: int
    max_seq_len: int

    def tokenize_batch(self, prompts: Sequence[str], device: str) -> torch.Tensor:
        ...

    def forward_with_intervention(self, input_ids: torch.Tensor, values: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        ...


class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, nlayers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def forward(self, prog_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.emb(prog_ids))


class ACAHyperNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        program_vocab_size: int,
        num_layers: int,
        max_positions: int,
        intervention_rank: int,
        hyper_layers: int = 2,
        hyper_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_positions = max_positions
        self.encoder = ProgramEncoder(program_vocab_size, d_model, nhead=hyper_heads, nlayers=hyper_layers)

        self.layer_read_head = nn.Linear(d_model, num_layers)
        self.layer_write_head = nn.Linear(d_model, num_layers)
        self.position_write_head = nn.Linear(d_model, max_positions)

        self.gain_head = nn.Sequential(nn.Linear(d_model, intervention_rank), nn.ReLU(), nn.Linear(intervention_rank, d_model))
        self.bias_head = nn.Sequential(nn.Linear(d_model, intervention_rank), nn.ReLU(), nn.Linear(intervention_rank, d_model))

    def forward(self, prog_ids: torch.Tensor) -> InterventionSpec:
        h = self.encoder(prog_ids)
        layer_read = torch.argmax(self.layer_read_head(h), dim=-1)
        layer_write = torch.argmax(self.layer_write_head(h), dim=-1)
        position_write = torch.argmax(self.position_write_head(h), dim=-1)
        gain = self.gain_head(h)
        bias = self.bias_head(h)
        return InterventionSpec(
            layer_read=layer_read.reshape(-1),
            layer_write=layer_write.reshape(-1),
            position_write=position_write.reshape(-1),
            gain=gain.reshape(-1, self.d_model),
            bias=bias.reshape(-1, self.d_model),
        )


class HFActivationPatchingBackend(nn.Module):
    """Real activation-patching backend for OLMo/Llama-like decoder models."""

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype,
        max_seq_len: int,
        attn_implementation: str = "sdpa",
        device_map: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_seq_len = max_seq_len
        self.d_model = int(self.model.config.hidden_size)
        self.num_layers = int(self.model.config.num_hidden_layers)

    def _decoder_layers(self):
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise RuntimeError("Unsupported transformer architecture for layer hooks")

    def tokenize_batch(self, prompts: Sequence[str], device: str) -> torch.Tensor:
        t = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return t.input_ids.to(device)

    def forward_with_intervention(self, input_ids: torch.Tensor, values: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        layer_mods: Dict[int, List[Tuple[int, int, torch.Tensor]]] = {}
        bsz = input_ids.size(0)
        n_steps = spec.layer_write.numel() // bsz

        for flat_i in range(spec.layer_write.numel()):
            batch_i = flat_i // n_steps
            layer = int(spec.layer_write[flat_i].item())
            pos = int(spec.position_write[flat_i].item())
            patch = spec.gain[flat_i] * values[flat_i] + spec.bias[flat_i]
            layer_mods.setdefault(layer, []).append((batch_i, pos, patch))

        handles = []
        layers = self._decoder_layers()

        def make_hook(layer_idx: int):
            def hook(_module, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                if layer_idx not in layer_mods:
                    return out
                for bi, pos, vec in layer_mods[layer_idx]:
                    p = pos % hidden.size(1)
                    hidden[bi, p, :] = hidden[bi, p, :] + vec.to(hidden.dtype)
                if isinstance(out, tuple):
                    return (hidden,) + out[1:]
                return hidden

            return hook

        for l in layer_mods.keys():
            if 0 <= l < len(layers):
                handles.append(layers[l].register_forward_hook(make_hook(l)))

        logits = self.model(input_ids=input_ids).logits[:, -1, :]
        for h in handles:
            h.remove()
        return logits


class FrozenLLMStub(nn.Module):
    """Fast fallback backend for local sanity checks without model downloads."""

    def __init__(self, num_entities: int, d_model: int, num_layers: int, max_positions: int):
        super().__init__()
        self.num_entities = num_entities
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_positions
        self.token_emb = nn.Embedding(num_entities + 64, d_model)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU()) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, num_entities)
        for p in self.parameters():
            p.requires_grad = False

    def tokenize_batch(self, prompts: Sequence[str], device: str) -> torch.Tensor:
        max_len = 48
        toks = torch.zeros((len(prompts), max_len), dtype=torch.long, device=device)
        for i, p in enumerate(prompts):
            vals = [ord(c) % self.token_emb.num_embeddings for c in p[:max_len]]
            toks[i, : len(vals)] = torch.tensor(vals, dtype=torch.long, device=device)
        return toks

    def forward_with_intervention(self, input_ids: torch.Tensor, values: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        h = self.token_emb(input_ids)
        bsz = h.size(0)
        n_steps = spec.layer_write.numel() // bsz
        layer_mods: Dict[int, List[Tuple[int, int, torch.Tensor]]] = {}
        for flat_i in range(spec.layer_write.numel()):
            bi = flat_i // n_steps
            layer = int(spec.layer_write[flat_i].item()) % self.num_layers
            pos = int(spec.position_write[flat_i].item()) % h.size(1)
            patch = spec.gain[flat_i] * values[flat_i] + spec.bias[flat_i]
            layer_mods.setdefault(layer, []).append((bi, pos, patch))

        for l_idx, layer in enumerate(self.layers):
            h = h + layer(h)
            for bi, pos, vec in layer_mods.get(l_idx, []):
                h[bi, pos, :] = h[bi, pos, :] + vec.to(h.dtype)
        h = self.ln(h)
        return self.out(h[:, -1, :])


def make_prompt_for_example(context_triples: Sequence[Tuple[int, int, int]], question: str) -> str:
    triples = "; ".join([f"(e{s}, r{r}, e{o})" for s, r, o in context_triples])
    return f"Context: {triples}\nQuestion: {question}\nAnswer:"


def program_to_ids(program: Program, max_len: int, pad_id: int = 0) -> List[int]:
    ids = [int(step.primitive) + 1 for step in program.steps][:max_len]
    if len(ids) < max_len:
        ids.extend([pad_id] * (max_len - len(ids)))
    return ids


def primitive_histogram(programs: List[Program]) -> Dict[Primitive, int]:
    out = {p: 0 for p in Primitive}
    for prog in programs:
        for step in prog.steps:
            out[step.primitive] += 1
    return out
