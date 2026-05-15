from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from acar.baselines import StaticIITBaseline
from acar.config import DataConfig, EvalConfig, ModelConfig, TrainConfig
from acar.data import MultiHopGenerator
from acar.eval import evaluate_split, extract_rules
from acar.external_baselines import ExternalBaselineRunner
from acar.models import ACAHyperNetwork, HFActivationPatchingBackend, FrozenLLMStub
from acar.reporting import ensure_dirs, plot_accuracy, plot_emi, plot_faithfulness, write_json, write_metrics_table
from acar.training import ACATrainer, collate_examples
from acar.utils import set_seed


def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def train_hyper(
    llm,
    hyper,
    train_data,
    train_cfg: TrainConfig,
    num_entities: int,
    max_prog_len: int,
    d_model: int,
):
    trainer = ACATrainer(
        llm=llm,
        hyper=hyper,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        grad_clip=train_cfg.grad_clip,
        device=train_cfg.device,
    )
    for epoch in range(train_cfg.epochs):
        running = 0.0
        for i in range(0, len(train_data), train_cfg.batch_size):
            batch_examples = train_data[i : i + train_cfg.batch_size]
            b = collate_examples(
                batch_examples,
                llm=llm,
                num_entities=num_entities,
                max_prog_len=max_prog_len,
                d_model=d_model,
                device=train_cfg.device,
            )
            loss = trainer.train_step(b)
            running += loss
            step = (i // train_cfg.batch_size) + 1
            if step % train_cfg.log_every == 0:
                print(f"epoch={epoch+1} step={step} train_loss={running/train_cfg.log_every:.4f}")
                running = 0.0


def eval_method(method_name: str, llm, method_model, splits, data_cfg, eval_cfg, train_cfg, max_prog_len, d_model):
    rows = []
    for split_name, examples in splits.items():
        if split_name == "train":
            continue
        m = evaluate_split(
            llm=llm,
            hyper=method_model,
            examples=examples,
            num_entities=data_cfg.num_entities,
            max_prog_len=max_prog_len,
            d_model=d_model,
            corruption_scale=eval_cfg.corruption_scale,
            device=train_cfg.device,
        )
        rows.append({"method": method_name, "split": split_name, **m})
    return rows


def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-stub", action="store_true", help="Use local stub backend instead of OLMo/Llama")
    parser.add_argument("--llm-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-external-baselines", action="store_true")
    args = parser.parse_args()

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    eval_cfg = EvalConfig()

    if args.output_dir:
        eval_cfg.output_dir = args.output_dir

    if not torch.cuda.is_available():
        train_cfg.device = "cpu"
        model_cfg.dtype = "float32"

    set_seed(train_cfg.seed)
    paths = ensure_dirs(eval_cfg.output_dir)

    generator = MultiHopGenerator(
        num_entities=data_cfg.num_entities,
        relation_names=data_cfg.train_relations,
        test_relation_names=data_cfg.test_only_relations,
        min_edges=data_cfg.min_context_edges,
        max_edges=data_cfg.max_context_edges,
    )
    splits = generator.build_all_splits(
        train_size=data_cfg.train_size,
        iid_eval_size=data_cfg.iid_eval_size,
        length_eval_size=data_cfg.length_eval_size,
        relation_eval_size=data_cfg.relation_eval_size,
        novel_comp_eval_size=data_cfg.novel_comp_eval_size,
        max_hops_train=data_cfg.max_hops_train,
        max_hops_eval=data_cfg.max_hops_eval,
    )

    llm_name = args.llm_name or model_cfg.llm_name
    if args.use_stub:
        llm = FrozenLLMStub(
            num_entities=data_cfg.num_entities,
            d_model=512,
            num_layers=24,
            max_positions=model_cfg.max_seq_len,
        ).to(train_cfg.device)
        d_model = 512
        num_layers = 24
    else:
        llm = HFActivationPatchingBackend(
            model_name=llm_name,
            dtype=parse_dtype(model_cfg.dtype),
            max_seq_len=model_cfg.max_seq_len,
            attn_implementation=model_cfg.attn_implementation,
            device_map="auto" if train_cfg.device.startswith("cuda") else None,
        )
        d_model = llm.d_model
        num_layers = llm.num_layers

    max_prog_len = 16
    acar_model = ACAHyperNetwork(
        d_model=d_model,
        program_vocab_size=model_cfg.program_vocab_size,
        num_layers=num_layers,
        max_positions=model_cfg.max_seq_len,
        intervention_rank=model_cfg.intervention_rank,
        hyper_layers=model_cfg.hyper_layers,
    ).to(train_cfg.device)

    print("Training ACAR hypernetwork...")
    train_hyper(
        llm=llm,
        hyper=acar_model,
        train_data=splits["train"],
        train_cfg=train_cfg,
        num_entities=data_cfg.num_entities,
        max_prog_len=max_prog_len,
        d_model=d_model,
    )

    ckpt_path = paths["checkpoints"] / "acar_hyper.pt"
    torch.save(acar_model.state_dict(), ckpt_path)

    rules = extract_rules(acar_model, splits["iid"][: eval_cfg.num_rule_extraction_samples], max_prog_len=max_prog_len, device=train_cfg.device)
    write_json(
        paths["tables"] / "rewrite_rules.json",
        {
            "llm": llm_name if not args.use_stub else "stub",
            "rules": [
                {
                    "primitive": r.primitive.name,
                    "layer_read": r.layer_read_mode,
                    "layer_write": r.layer_write_mode,
                    "position_write": r.position_write_mode,
                }
                for r in rules
            ],
        },
    )

    rows = []
    rows.extend(eval_method("ACAR", llm, acar_model, splits, data_cfg, eval_cfg, train_cfg, max_prog_len, d_model))

    # Static IIT baseline (non-amortized)
    static_iit = StaticIITBaseline(max_prog_len=max_prog_len, d_model=d_model, num_layers=num_layers, max_positions=model_cfg.max_seq_len).to(train_cfg.device)
    rows.extend(eval_method("StaticIIT", llm, static_iit, splits, data_cfg, eval_cfg, train_cfg, max_prog_len, d_model))

    if args.include_external_baselines:
        ext = ExternalBaselineRunner()
        for split_name, examples in splits.items():
            if split_name == "train":
                continue
            rows.append(ext.run_acdc(split_name, examples).__dict__)
            rows.append(ext.run_tokenshap(split_name, examples).__dict__)
            rows.append(ext.run_cot_faithfulness(split_name, examples).__dict__)

    df = write_metrics_table(paths["tables"] / "metrics.csv", rows)
    write_json(paths["tables"] / "metrics.json", {"rows": rows})

    plot_faithfulness(df, paths["figures"] / "faithfulness.png")
    plot_emi(df, paths["figures"] / "emi.png")
    plot_accuracy(df, paths["figures"] / "accuracy.png")

    summary = {
        "output_dir": str(paths["root"]),
        "checkpoint": str(ckpt_path),
        "metrics_csv": str(paths["tables"] / "metrics.csv"),
        "plots": ["faithfulness.png", "emi.png", "accuracy.png"],
        "splits": {k: len(v) for k, v in splits.items()},
        "backend": "stub" if args.use_stub else llm_name,
    }
    write_json(paths["root"] / "run_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
