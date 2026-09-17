# `cod` package

Query-time evidence organization on raw conversational memory.

| File | Role |
|------|------|
| `store.py` | Fidelity-preserving utterance index + dense retrieval |
| `selector.py` | **CoD** one-shot greedy seed allocation (`R+λD+βC`) |
| `gap_agent.py` | **Gap-Directed Agent** (iterative gap closing) |
| `sde.py` | Seed-then-Expand primitive / baseline |
| `controller.py` | IMR LLM-controller baseline (negative control) |
| `run.py` | Baselines: `raw_dense`, `raw_temporal`, `sde`, `imr` |
| `run_cod.py` | Full LoCoMo CoD evaluation |
| `run_gap_agent.py` | Gap-Directed selection vs CoD / dense |

See the repository root `README.md` for commands.
