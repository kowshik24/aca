from dataclasses import dataclass


@dataclass
class DataConfig:
    num_entities: int = 40
    train_relations: tuple[str, ...] = ("taller", "older")
    test_only_relations: tuple[str, ...] = ("heavier",)
    min_context_edges: int = 90
    max_context_edges: int = 180
    max_hops_train: int = 3
    max_hops_eval: int = 6
    train_size: int = 4000
    iid_eval_size: int = 500
    length_eval_size: int = 500
    relation_eval_size: int = 500
    novel_comp_eval_size: int = 500


@dataclass
class ModelConfig:
    llm_name: str = "allenai/OLMo-7B-0724-hf"
    fallback_llm_name: str = "meta-llama/Meta-Llama-3-8B"
    use_real_llm: bool = True
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    max_seq_len: int = 384
    d_model: int = 4096
    program_vocab_size: int = 256
    hyper_hidden: int = 512
    hyper_layers: int = 2
    intervention_rank: int = 64
    use_low_rank_transform: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 3
    grad_clip: float = 1.0
    device: str = "cuda"
    log_every: int = 25


@dataclass
class EvalConfig:
    corruption_scale: float = 2.0
    num_rule_extraction_samples: int = 600
    output_dir: str = "results"
