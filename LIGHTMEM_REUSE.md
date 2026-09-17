# LightMem reuse note

This package does **not** depend on the LightMem Python package at runtime.

## Prompt / judge reuse (exact)

These files are **byte-identical** to the corresponding LightMem LoCoMo experiment files:

| File in this package | LightMem counterpart | Status |
|----------------------|----------------------|--------|
| `llm_judge.py` | `experiments/locomo/llm_judge.py` | identical |
| `cod/prompts.py` | `experiments/locomo/ours1/prompts.py` | identical |

### Prompts reused verbatim (7 / 7 = 100%)

From `llm_judge.py`:

- `ACCURACY_PROMPT`

From `cod/prompts.py`:

- `CONTROLLER_SYSTEM_PROMPT`
- `CONTROLLER_USER_PROMPT`
- `ANSWER_SYSTEM_PROMPT`
- `ANSWER_USER_PROMPT`
- `REASONING_PATCH_SYSTEM_PROMPT`
- `REASONING_PATCH_USER_PROMPT`

## Not reused as a runtime dependency

- LightMem memory engine / `structmem` baseline (omitted).
- CoD selector and Gap-Directed Agent are original to this codebase.
- Embedding / store path is standalone MiniLM + Qdrant.

| Category | Reuse |
|----------|--------|
| Judge accuracy prompt | 100% identical |
| IMR / answer / reasoning prompts | 100% identical (all 6 strings) |
| CoD selector + Gap-Directed Agent | original |
| LoCoMo JSON | public benchmark |
