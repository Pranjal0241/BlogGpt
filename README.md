# BlogGpt

A LangGraph pipeline that turns a topic into a reviewed blog post.

The interesting part is the review loop: drafted sections are scored by a
critic, weak ones are sent back to a rewriter with the critic's notes, and the
loop is bounded so runs always terminate.

```
START -> brief -> outline -> fanout(draft) -> review -> figure -> assemble -> END
                                          ^         |
                                          +-- revise+
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then add one API key
```

## Run

```bash
streamlit run bwa_frontend.py
```

Or from Python:

```python
from bwa_backend import write_post

print(write_post("why vector databases lose to plain Postgres"))
```

## Tests

```bash
python test_pipeline.py
```

Runs entirely offline against a stub model — no API key, no network. Covers the
clean path, the revision path, and the case where a section never satisfies the
critic, which must still terminate at `MAX_REVISIONS`.

## How it works

| Node | Job |
| --- | --- |
| `brief` | Editorial brief: audience, prior knowledge, the promise of the post |
| `outline` | Structured outline; each section carries an `intent`, points and a word target |
| `draft` | Fanned out, one model call per section, run concurrently |
| `review` | Scores each section 0-1 and names the weakest claim |
| `revise` | Rewrites only the failing sections using the critic's findings |
| `figure` | Proposes one technical diagram |
| `assemble` | Joins sections in order and writes the markdown |

### Design notes

**State channels are dicts, not lists.** `sections` and `critiques` are keyed
maps. Concurrent draft workers therefore cannot overwrite each other, and a
revision pass replaces exactly one section without disturbing its siblings — an
append-only list cannot express that.

**Structured output uses native tool calling.** `with_structured_output` makes
the provider validate the response against the schema, so malformed JSON never
reaches the graph.

**The review loop is bounded two ways.** `MAX_REVISIONS` caps passes per
section, and `REVIEW_THRESHOLD` decides what counts as failing. A bad run
degrades into "unrevised output" rather than an infinite loop. The budget is
checked per section, so a section drafted weak on its first pass is still
eligible for a rewrite.

**Provider content shapes are normalised in one place.** `as_text()` flattens
OpenAI's `str` content and Gemini's list-of-parts into plain text, so nodes stay
provider-agnostic.

**Model selection is pluggable.** `get_model()` prefers OpenAI, falls back to
OpenRouter, then Gemini, and raises a clear error when nothing is configured.

## Tuning

| Variable | Default | Effect |
| --- | --- | --- |
| `REVIEW_THRESHOLD` | `0.75` | Sections scoring lower get rewritten |
| `MAX_REVISIONS` | `2` | Cap on rewrite passes per section |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Also used for diagram generation |