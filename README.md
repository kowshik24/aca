# ACAR

Amortized Causal Abstraction with compositional rewrite rules for multi-hop reasoning analysis in LLMs.

## What this repo includes

- Synthetic relational reasoning dataset and symbolic DSL traces.
- ACAR hypernetwork that predicts activation interventions from program steps.
- Real Hugging Face activation patching backend for OLMo/Llama.
- Static IIT baseline and evaluation across:
  - IID
  - Length generalization
  - Relation generalization
  - Novel primitive composition
- Result artifact generation (metrics tables + plots).

## Setup

```bash
pip install -e .
```

## Run

Full run (real model):

```bash
python scripts/run_full_acar.py --llm-name allenai/OLMo-7B-0724-hf --output-dir results
```

Fallback model:

```bash
python scripts/run_full_acar.py --llm-name meta-llama/Meta-Llama-3-8B --output-dir results
```

Local sanity run (no large model download):

```bash
python scripts/run_full_acar.py --use-stub --output-dir results_stub
```

## Outputs

- `results/tables/metrics.csv`
- `results/tables/metrics.json`
- `results/tables/rewrite_rules.json`
- `results/figures/faithfulness.png`
- `results/figures/emi.png`
- `results/figures/accuracy.png`
- `results/checkpoints/acar_hyper.pt`
- `results/run_summary.json`
