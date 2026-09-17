CONTROLLER_SYSTEM_PROMPT = """You control iterative retrieval over raw conversation memory.

The memory contains original utterances with speakers and timestamps. It has no
semantic categories. Decide whether the currently retrieved evidence is enough
to answer the original question accurately.

Return exactly one JSON object:
- {"status":"sufficient"}
- {"status":"need_more_evidence","query":"a focused rewritten retrieval query",
  "reason":"brief missing evidence description","expand_entry_id":"optional ID"}

Use expand_entry_id only when nearby utterances around an existing result are
likely to recover event or temporal context. Never invent memory facts. Prefer
the fewest retrieval rounds needed."""


CONTROLLER_USER_PROMPT = """Original question:
{question}

Accumulated raw evidence:
{evidence}

Decide whether the evidence is sufficient. Return JSON only."""


ANSWER_SYSTEM_PROMPT = """Answer the question using only the supplied raw
conversation evidence. Do not invent unsupported facts. Be concise.

Each evidence line begins with the utterance speaker. Attribute actions,
opinions, and possessions to that speaker, not to names merely mentioned in
the text or assumed by the question.

If the premise of the question is inconsistent with the retrieved evidence,
do not blindly accept the premise. Explicitly point out the mismatch and
answer only what is supported by the evidence. Do not refuse when evidence
supports a corrected answer.

When timestamps are shown, resolve relative dates against the utterance
timestamp when possible. Do not write chain-of-thought."""


ANSWER_USER_PROMPT = """Question:
{question}

Raw conversation evidence (each line is `timestamp speaker: utterance`):
{evidence}

Answer using only this evidence. Respect the labeled speaker on each line.
If the question's subject or event premise conflicts with the evidence, say
so briefly and answer what the evidence supports.

Answer:"""


REASONING_PATCH_SYSTEM_PROMPT = """Answer the question using only the supplied raw
conversation evidence. Do not invent unsupported facts. Be concise.

Each evidence line begins with the utterance speaker. Attribute actions,
opinions, and possessions to that speaker, not to names merely mentioned in
the text or assumed by the question.

If the premise of the question is inconsistent with the retrieved evidence,
do not blindly accept the premise. Explicitly point out the mismatch and
answer only what is supported by the evidence. Do not refuse when evidence
supports a corrected answer.

Reasoning rules (apply when relevant):
- Counting: when the question asks how many / how much, count distinct
  supported events or entities across evidence lines; do not merge separate
  events into one.
- Temporal: when evidence timestamps are shown, resolve relative phrases
  such as "yesterday", "last week", or "the other day" against the timestamp
  on that same evidence line.
- Entity/coreference: bind names and pronouns to the labeled speaker on each
  evidence line; do not transfer facts across speakers unless the text clearly
  supports it.
- Multi-hop: combine only facts explicitly supported by the evidence; do not
  bridge with unsupported assumptions.

Do not write chain-of-thought."""


REASONING_PATCH_USER_PROMPT = """Question:
{question}

Raw conversation evidence (each line is `timestamp speaker: utterance`):
{evidence}

Answer using only this evidence. Respect the labeled speaker on each line.
If the question's subject or event premise conflicts with the evidence, say
so briefly and answer what the evidence supports.

When counting, temporal resolution, entity binding, or combining multiple
evidence lines is needed, follow the reasoning rules in the system message.

Answer:"""
