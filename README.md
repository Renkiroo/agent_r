# Chain on Demand (CoD) + Gap-Directed Agent (LoCoMo)

Anonymous research code for **query-time evidence organization** on long conversational memory.

Two deployable modes share the same fidelity-preserving raw-utterance index:

1. **CoD (Chain on Demand)** — one-shot greedy seed allocation with conditioned scoring, then deferred temporal expansion.
2. **Gap-Directed Agent** — iterative gap closing over the same candidate pool and budget.

This folder is self-contained. Upload **this directory only**.

## Contents

| Path | Description |
|------|-------------|
| `cod/` | Method package: store, CoD selector, Gap-Directed agent, baselines |
| `data/` | LoCoMo-10, 100-question subset, full QA list |
| `qdrant_raw/` | Pre-built utterance embeddings |
| `config.yaml` | Default 100-question subset |
| `config.cod.yaml` | Full LoCoMo CoD run |
| `scripts/` | Setup check + smokes |
| `llm_judge.py` | LoCoMo LLM judge |
| `LIGHTMEM_REUSE.md` | Prompt provenance vs LightMem |

## Quick start

```bash
pip install -r requirements.txt
cp config.local.example.yaml config.local.yaml
# Edit openai_api_key / openai_api_base

export PYTHONPATH=.          # Windows: set PYTHONPATH=.
python scripts/verify_setup.py
```

### Smoke tests

```bash
# Selection only — no API key
python scripts/smoke_gap_agent.py --config config.yaml

# Full QA — CoD smoke (needs API)
python scripts/smoke_cod.py --config config.yaml
```

### Baselines

```bash
python -m cod.run --method raw_temporal --config config.yaml
python -m cod.run --method sde --config config.yaml
python -m cod.run --method imr --config config.yaml   # LLM controller (negative control)
```

### CoD (main method)

```bash
python -m cod.run_cod --config config.cod.yaml --workers 3
```

### Gap-Directed Agent (main agent)

```bash
python -m cod.run_gap_agent --config config.yaml --output-dir results/gap_agent
```

## Frozen hyperparameters

`λ=0.06`, `β=0.10`, pool `K=200`, seeds `K=60`, expand window `±2`.

## Tests

```bash
python -m pytest cod/tests/ -q
```

## Architecture

```text
LoCoMo QA + qdrant_raw
        ↓
RawMemoryStore (dense + temporal expand)
        ↓
   CoD (one-shot)     or     Gap-Directed Agent
        ↓
LLM answer + judge (optional)
```

See `cod/README.md` for the module map.
