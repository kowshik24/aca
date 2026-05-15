# Colab Runbook (ACAR)

## 1) Runtime

- GPU: A100/L4 recommended.
- Python: 3.10+.

## 2) Setup

```bash
git clone <your-repo-url>
cd aca
pip install -e .
```

## 3) Full ACAR run (real model)

```bash
python scripts/run_full_acar.py \
  --llm-name allenai/OLMo-7B-0724-hf \
  --output-dir results
```

Fallback:

```bash
python scripts/run_full_acar.py \
  --llm-name meta-llama/Meta-Llama-3-8B \
  --output-dir results
```

## 4) Include blueprint external-baseline placeholders in output table

```bash
python scripts/run_full_acar.py \
  --llm-name allenai/OLMo-7B-0724-hf \
  --output-dir results \
  --include-external-baselines
```

## 5) Regenerate plots from metrics

```bash
python scripts/make_paper_figures.py \
  --metrics results/tables/metrics.csv \
  --fig-dir results/figures
```

## 6) Expected output folders

- `results/tables`
- `results/figures`
- `results/checkpoints`
- `results/run_summary.json`
