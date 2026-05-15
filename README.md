# ACAR (Amortized Causal Abstraction with Compositional Rewrite Rules)

Blueprint-complete research codebase for:
- DSL + symbolic trace generation
- Real OLMo/Llama activation patching backend
- ACA hypernetwork IIT training
- Rewrite rule extraction
- Split-wise evaluation (IID, length, relation, novel composition)
- Artifact generation (tables + images)

## Repository layout

- `acar/data.py`: synthetic relational dataset + official split builders.
- `acar/dsl.py`: DSL primitives and symbolic executor.
- `acar/models.py`: ACA hypernetwork + real HF activation patching backend.
- `acar/training.py`: IIT training loop.
- `acar/eval.py`: faithfulness, EMI proxy, rule extraction, split metrics.
- `acar/baselines.py`: static IIT baseline.
- `acar/reporting.py`: metrics tables and plotting utilities.
- `scripts/run_full_acar.py`: full experiment runner.
- `scripts/make_paper_figures.py`: regenerate paper figures from metrics.

## Install

```bash
pip install -e .
```

## Quick local sanity run (no model download)

```bash
python scripts/run_full_acar.py --use-stub --output-dir results_stub
```

## Full OLMo/Llama run (Colab/GPU)

```bash
python scripts/run_full_acar.py \
  --llm-name allenai/OLMo-7B-0724-hf \
  --output-dir results
```

If OLMo access is restricted in your environment, use Llama:

```bash
python scripts/run_full_acar.py \
  --llm-name meta-llama/Meta-Llama-3-8B \
  --output-dir results
```

## Outputs

`results/`
- `tables/metrics.csv`
- `tables/metrics.json`
- `tables/rewrite_rules.json`
- `figures/faithfulness.png`
- `figures/emi.png`
- `figures/accuracy.png`
- `checkpoints/acar_hyper.pt`
- `run_summary.json`

## Notes on blueprint coverage

Implemented directly from `BLUE_PRINT.md`:
- Amortized hypernetwork over symbolic programs
- Activation intervention parameters (`layer_read/layer_write/position + gain/bias`)
- Rule extraction by primitive-wise modal clustering
- Counterfactual faithfulness and sufficiency (EMI proxy)
- Generalisation split suite

Current limitation:
- ACDC and TokenSHAP/CoT baselines are scaffold targets for next extension; static IIT baseline is included and evaluated now.
