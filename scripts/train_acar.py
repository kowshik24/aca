from __future__ import annotations

import argparse

import torch

from acar.config import DataConfig, EvalConfig, ModelConfig, TrainConfig
from acar.data import MultiHopGenerator
from acar.eval import counterfactual_faithfulness, extract_rules
from acar.models import ACAHyperNetwork, FrozenLLMStub
from acar.training import ACATrainer, collate_examples
from acar.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    eval_cfg = EvalConfig()

    if args.device is not None:
        train_cfg.device = args.device

    set_seed(train_cfg.seed)

    generator = MultiHopGenerator(
        num_entities=data_cfg.num_entities,
        relations=list(range(data_cfg.num_relations)),
        min_edges=data_cfg.min_context_edges,
        max_edges=data_cfg.max_context_edges,
    )

    train_data = generator.sample(train_cfg.steps_per_epoch * train_cfg.batch_size, max_hops=data_cfg.max_hops_train)
    eval_data = generator.sample(eval_cfg.num_eval_samples, max_hops=data_cfg.max_hops_eval)

    llm = FrozenLLMStub(
        num_entities=data_cfg.num_entities,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        max_positions=model_cfg.max_positions,
    )
    hyper = ACAHyperNetwork(
        d_model=model_cfg.d_model,
        program_vocab_size=model_cfg.program_vocab_size,
        num_layers=model_cfg.num_layers,
        max_positions=model_cfg.max_positions,
        intervention_rank=model_cfg.intervention_rank,
    )

    trainer = ACATrainer(llm=llm, hyper=hyper, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay, device=train_cfg.device)

    max_prog_len = 12
    for epoch in range(train_cfg.epochs):
        running = 0.0
        for step in range(train_cfg.steps_per_epoch):
            start = step * train_cfg.batch_size
            end = start + train_cfg.batch_size
            batch = collate_examples(
                train_data[start:end],
                num_entities=data_cfg.num_entities,
                max_prog_len=max_prog_len,
                d_model=model_cfg.d_model,
                max_positions=model_cfg.max_positions,
            )
            loss = trainer.train_step(batch)
            running += loss
            if (step + 1) % train_cfg.log_every == 0:
                print(f"epoch={epoch+1} step={step+1} train_loss={running/train_cfg.log_every:.4f}")
                running = 0.0

        eb = collate_examples(
            eval_data[: train_cfg.batch_size],
            num_entities=data_cfg.num_entities,
            max_prog_len=max_prog_len,
            d_model=model_cfg.d_model,
            max_positions=model_cfg.max_positions,
        )
        ev_loss, ev_acc = trainer.eval_step(eb)
        print(f"epoch={epoch+1} eval_loss={ev_loss:.4f} eval_acc={ev_acc:.4f}")

    rules = extract_rules(hyper, eval_data[:64], max_prog_len=max_prog_len, device=train_cfg.device)
    print("\nExtracted rewrite rules (mode stats):")
    for rule in rules:
        print(
            f"primitive={rule.primitive.name} layer_read={rule.layer_read_mode} "
            f"layer_write={rule.layer_write_mode} pos_write={rule.position_write_mode}"
        )

    faith = counterfactual_faithfulness(
        llm=llm,
        hyper=hyper,
        examples=eval_data[:128],
        num_entities=data_cfg.num_entities,
        max_prog_len=max_prog_len,
        d_model=model_cfg.d_model,
        max_positions=model_cfg.max_positions,
        corruption_scale=eval_cfg.corruption_scale,
        device=train_cfg.device,
    )
    print(f"\nCounterfactual faithfulness={faith:.4f}")


if __name__ == "__main__":
    main()
