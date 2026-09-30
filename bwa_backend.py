"""BlogGpt - a LangGraph pipeline that turns a topic into a reviewed blog post.

Architecture
------------
    START -> brief -> outline -> fanout(draft) -> review -> figure -> assemble -> END
                                          ^         |
                                          +-- revise+

Design notes
------------
The distinguishing idea here is the review/revise loop. A single-pass
generator produces mediocre prose because nothing checks it. So every drafted
section is scored by a critic node, weak sections are bounced back to a
rewriter with the critic's notes, and the loop is bounded by MAX_REVISIONS so
a run always terminates.

Sections are keyed and written to dict-backed state channels. Concurrent draft
workers therefore never clobber each other, and a revision pass can replace
exactly one section without disturbing its siblings - which an append-only
list cannot do.

Structured output is obtained with native tool calling rather than prompt
engineering into JSON, so the provider validates the response for us.
"""

from __future__ import annotations

import operator
import os
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv()

MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.75"))


# --------------------------------------------------------------------------
# Structured contracts
# --------------------------------------------------------------------------
class Section(BaseModel):
    key: str = Field(description="Stable slug used as the state key, e.g. 'why-rag'.")
    heading: str
    intent: str = Field(description="What the reader should be able to do after this section.")
    points: list[str] = Field(min_length=2, max_length=5)
    target_words: int = Field(ge=120, le=700)


class Outline(BaseModel):
    title: str
    subtitle: str = ""
    audience: str
    angle: str = Field(description="The one claim that makes this post worth reading.")
    sections: list[Section] = Field(min_length=3, max_length=7)


class Critique(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="1.0 is publishable.")
    verdict: Literal["pass", "revise"]
    findings: list[str] = Field(default_factory=list, max_length=4)
    weakest_claim: str = ""


class ArtPlan(BaseModel):
    figure_after_section: str = Field(description="Section key the figure follows.")
    caption: str
    prompt: str = Field(description="Text-to-image prompt, technical diagram style.")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
Sections = Annotated[dict[str, str], operator.or_]
Critiques = Annotated[dict[str, Critique], operator.or_]


class State(TypedDict):
    topic: str
    voice: str
    as_of: str

    outline: Outline | None
    sections: Sections
    critiques: Critiques
    revisions: Annotated[dict[str, int], operator.or_]

    figure: ArtPlan | None
    figure_path: str
    document: str
    notes: list[str]


# --------------------------------------------------------------------------
# Model access
# --------------------------------------------------------------------------
def _looks_configured(value: str) -> bool:
    value = (value or "").strip()
    return len(value) > 20 and "your-" not in value and "placeholder" not in value


def get_model(kind: Literal["text", "image"] = "text"):
    """Return a chat model, preferring OpenAI, then OpenRouter, then Gemini."""
    if _looks_configured(os.getenv("OPENAI_API_KEY", "")):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0.7,
        )

    if _looks_configured(os.getenv("OPENROUTER_API_KEY", "")):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5-air:free"),
            temperature=0.7,
        )

    if _looks_configured(os.getenv("GOOGLE_API_KEY", "")):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.7,
        )

    raise RuntimeError(
        "No LLM configured. Set one of OPENAI_API_KEY, OPENROUTER_API_KEY or "
        "GOOGLE_API_KEY in .env (see .env.example)."
    )


def as_text(message: Any) -> str:
    """Flatten provider-specific content shapes into plain text.

    OpenAI returns a str; Gemini returns a list of typed parts. Handling both
    here keeps every downstream node provider-agnostic.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
            elif getattr(part, "text", None):
                chunks.append(str(part.text))
        return "\n".join(chunks).strip()
    return str(content).strip()


def ask(prompt: str, system: str, schema: type[BaseModel] | None = None) -> Any:
    """Single model call. Returns a validated `schema` when one is given."""
    model = get_model()
    if schema is not None:
        model = model.with_structured_output(schema)
    reply = model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    return reply if schema is not None else as_text(reply)


# --------------------------------------------------------------------------
# Node 1 - brief
# --------------------------------------------------------------------------
BRIEF_SYSTEM = """You are a senior editor opening a new technical publication.

Given a raw topic, produce a one-paragraph editorial brief: who this is for,
what they already know, and what promise the post must keep. Be concrete.
Return only the brief paragraph, no headings."""


def brief_node(state: State) -> dict[str, Any]:
    brief = ask(
        f"Topic: {state['topic']}\nVoice: {state['voice']}\nAs of: {state['as_of']}",
        BRIEF_SYSTEM,
    )
    return {"notes": [f"BRIEF: {brief}"]}


# --------------------------------------------------------------------------
# Node 2 - outline
# --------------------------------------------------------------------------
OUTLINE_SYSTEM = """You are a senior technical writer planning a blog post.

Produce an outline whose sections each carry real work. Rules:
- Sections must progress; no filler sections.
- `intent` states what the reader can DO afterwards, not what they will read.
- `points` are the sub-arguments, in order.
- `target_words` between 150 and 450. Longer sections must earn it.
Return strictly the outline."""


def outline_node(state: State) -> dict[str, Any]:
    prompt = (
        f"Topic: {state['topic']}\nVoice: {state['voice']}\nAs of: {state['as_of']}\n\n"
        "Write the outline."
    )
    outline = ask(prompt, OUTLINE_SYSTEM, Outline)
    return {"outline": outline, "notes": [f"OUTLINE: {len(outline.sections)} sections"]}


# --------------------------------------------------------------------------
# Node 3 - draft (fanned out, one call per section)
# --------------------------------------------------------------------------
DRAFT_SYSTEM = """You write ONE section of a technical blog post.

- Open with the point, not throat-clearing. No "in this article".
- Address the reader as 'you'.
- Explain the mechanism, not just the benefit. Show why it works.
- Use a concrete example or snippet when it clarifies.
- Close by tying back to the section's intent.
Markdown only. Do not restate the heading."""


def draft_node(payload: dict[str, Any]) -> dict[str, Any]:
    section = Section(**payload["section"])
    outline = Outline(**payload["outline"])

    bullets = "\n".join(f"- {point}" for point in section.points)
    prompt = (
        f"Post title: {outline.title}\n"
        f"Angle: {outline.angle}\n"
        f"Audience: {outline.audience}\n"
        f"Voice: {payload['voice']}\n\n"
        f"Section heading: {section.heading}\n"
        f"Intent: {section.intent}\n"
        f"Target length: about {section.target_words} words\n"
        f"Required points:\n{bullets}\n\n"
        f"Revision notes from the critic:\n{payload.get('findings') or '(none - first draft)'}"
    )

    body = ask(prompt, DRAFT_SYSTEM)
    return {"sections": {section.key: body}}


def fanout_drafts(state: State) -> list[Send]:
    outline = state["outline"]
    assert outline is not None
    return [
        Send(
            "draft",
            {
                "section": section.model_dump(),
                "outline": outline.model_dump(),
                "voice": state["voice"],
            },
        )
        for section in outline.sections
    ]


# --------------------------------------------------------------------------
# Node 4 - review
# --------------------------------------------------------------------------
REVIEW_SYSTEM = """You are a demanding line editor. You are reviewing one section.

Judge it against three bars, in order of importance:
1. Accuracy - is every claim defensible, or is it hand-waving?
2. Substance - does the mechanism get explained, or only the benefit?
3. Clarity - would a smart reader still be confused?

Be terse and specific. Name the exact sentence or claim that fails.
Score 0.0-1.0. Use verdict 'revise' below 0.75."""


def review_node(state: State) -> dict[str, Any]:
    outline = state["outline"]
    assert outline is not None
    critiques: dict[str, Critique] = {}

    for key, body in state["sections"].items():
        heading = next((s.heading for s in outline.sections if s.key == key), key)
        prompt = (
            f"Post: {outline.title}\nAngle: {outline.angle}\n"
            f"Section: {heading}\n\n---\n{body}\n---\n\nReview it."
        )
        critiques[key] = ask(prompt, REVIEW_SYSTEM, Critique)

    return {"critiques": critiques}


def failing(state: State) -> list[str]:
    """Sections the critic rejected that still have revision budget left.

    The budget is checked per section rather than across the run: a section
    that has not been revised yet has a count of zero, so a freshly drafted
    weak section is still eligible.
    """
    return [
        key
        for key, critique in state["critiques"].items()
        if (critique.verdict == "revise" or critique.score < REVIEW_THRESHOLD)
        and state["revisions"].get(key, 0) < MAX_REVISIONS
    ]


def needs_revision(state: State) -> str:
    """Bounce weak sections back for one more pass."""
    return "revise" if failing(state) else "assemble"


def revise_node(state: State) -> dict[str, Any]:
    """Rewriter: consumes the critic's findings for each failing section."""
    outline = state["outline"]
    assert outline is not None

    counts = state["revisions"]
    revised: dict[str, str] = {}
    updated: dict[str, int] = {}
    report: list[str] = []

    for key in failing(state):
        critique = state["critiques"][key]
        section = next(s for s in outline.sections if s.key == key)
        findings = "\n".join(f"- {item}" for item in critique.findings) or "- unspecific"

        prompt = (
            f"Post: {outline.title}\nSection: {section.heading}\n"
            f"Intent: {section.intent}\nVoice: {state['voice']}\n\n"
            f"Draft:\n{state['sections'][key]}\n\n"
            f"The critic scored this {critique.score:.2f} and said:\n{findings}\n"
            f"Weakest claim: {critique.weakest_claim or '(unspecified)'}\n\n"
            "Rewrite it. Fix every finding. Keep what already works."
        )
        revised[key] = ask(prompt, DRAFT_SYSTEM)
        updated[key] = counts.get(key, 0) + 1
        report.append(f"REVISED {key}: {critique.score:.2f} -> pending")

    return {"sections": revised, "revisions": updated, "notes": report}


# --------------------------------------------------------------------------
# Node 5 - figure + assemble
# --------------------------------------------------------------------------
FIGURE_SYSTEM = """You propose ONE technical diagram for a blog post.

Pick the single section most improved by a figure - architecture, data flow, or
a before/after comparison. Avoid decorative or conceptual art.

Return the section key, a caption, and a text-to-image prompt describing a
clean labelled technical diagram on a light background."""


def figure_node(state: State) -> dict[str, Any]:
    outline = state["outline"]
    assert outline is not None
    try:
        return {"figure": ask(f"Post: {outline.title}\n\nPropose the figure.", FIGURE_SYSTEM, ArtPlan)}
    except Exception:
        return {"figure": None}


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "blog-post"


def render_figure(prompt: str, destination: Path) -> None:
    """Generate the diagram with Gemini image models."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    for part in getattr(response, "parts", []) or []:
        blob = getattr(part, "inline_data", None)
        if blob and getattr(blob, "data", None):
            destination.write_bytes(blob.data)
            return

    for candidate in getattr(response, "candidates", []) or []:
        for part in candidate.content.parts:
            blob = getattr(part, "inline_data", None)
            if blob and getattr(blob, "data", None):
                destination.write_bytes(blob.data)
                return

    raise RuntimeError("Image model returned no image data.")


def assemble_node(state: State) -> dict[str, Any]:
    outline = state["outline"]
    assert outline is not None

    parts = [f"# {outline.title}"]
    if outline.subtitle:
        parts += ["", f"_{outline.subtitle}_"]

    figure_after = state["figure"].figure_after_section if state["figure"] else None
    figure_name = f"{slugify(outline.title)}-diagram.png"

    for section in outline.sections:
        parts += ["", f"## {section.heading}", "", state["sections"].get(section.key, "").strip()]

        if figure_after == section.key and state["figure"] is not None:
            if Path(figure_name).exists():
                parts += ["", f"![{state['figure'].caption}]({figure_name})", f"*{state['figure'].caption}*"]
            else:
                parts += ["", f"> _Figure unavailable: {state['figure'].caption}_"]

    parts += [
        "",
        "---",
        "",
        f"_Assembled {state['as_of']} · "
        + ", ".join(
            f"{key} v{count + 1}" for key, count in sorted(state["revisions"].items())
        )
        + (" · no revisions needed" if not state["revisions"] else "")
        + "_",
    ]

    document = "\n".join(parts).strip() + "\n"
    return {"document": document, "figure_path": figure_name if figure_after else ""}


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
builder = StateGraph(State)

builder.add_node("brief", brief_node)
builder.add_node("outline", outline_node)
builder.add_node("draft", draft_node)
builder.add_node("review", review_node)
builder.add_node("revise", revise_node)
builder.add_node("figure", figure_node)
builder.add_node("assemble", assemble_node)

builder.add_edge(START, "brief")
builder.add_edge("brief", "outline")
builder.add_conditional_edges("outline", fanout_drafts, ["draft"])
builder.add_edge("draft", "review")

# Review either loops back through the rewriter or moves on.
builder.add_conditional_edges(
    "review",
    needs_revision,
    {"revise": "revise", "assemble": "figure"},
)
builder.add_edge("revise", "review")

builder.add_edge("figure", "assemble")
builder.add_edge("assemble", END)

graph = builder.compile()


def write_post(topic: str, voice: str = "practical, no fluff", as_of: str | None = None) -> str:
    """Run the pipeline end to end and return the finished markdown."""
    result = graph.invoke(
        {
            "topic": topic,
            "voice": voice,
            "as_of": as_of or date.today().isoformat(),
            "outline": None,
            "sections": {},
            "critiques": {},
            "revisions": {},
            "figure": None,
            "figure_path": "",
            "document": "",
            "notes": [],
        }
    )
    return result["document"]