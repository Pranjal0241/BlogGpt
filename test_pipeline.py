"""Offline check of the graph: no API keys, no network.

Patches bwa_backend.ask so every model call is deterministic, then asserts the
pipeline produces a document, that the critic actually fires, and that the
revision cap holds.

Run: python test_pipeline.py
"""

from __future__ import annotations

import bwa_backend as backend
from bwa_backend import ArtPlan, Critique, Outline, Section, graph

call_log: list[str] = []


def outline_fixture() -> Outline:
    return Outline(
        title="Retrieval without the hype",
        subtitle="What vector search actually buys you",
        audience="backend engineers who have been told they need a vector database",
        angle="Similarity search is a trade, not an upgrade.",
        sections=[
            Section(key="annoyances", heading="What it fixes", intent="judge whether you need it",
                    points=["parametric staleness", "private data"], target_words=180),
            Section(key="mechanics", heading="How it works", intent="explain embedding recall",
                    points=["vectors", "distance"], target_words=220),
            Section(key="alternatives", heading="What else works", intent="compare against plain SQL",
                    points=["full-text", "pgvector"], target_words=200),
        ],
    )


def critique_fixture(sequence: list[str]) -> Critique:
    """Hands out the prepared verdicts, then passes."""
    verdict = sequence.pop(0) if sequence else "pass"
    return Critique(
        score=0.4 if verdict == "revise" else 0.9,
        verdict=verdict,
        findings=["too shallow"] if verdict == "revise" else [],
        weakest_claim="the second point" if verdict == "revise" else "",
    )


def install_stub(verdicts: list[str]) -> None:
    def fake_ask(prompt: str, system: str, schema=None):
        call_log.append(system.splitlines()[0][:48])
        if schema is Outline:
            return outline_fixture()
        if schema is Critique:
            return critique_fixture(verdicts)
        if schema is ArtPlan:
            return ArtPlan(figure_after_section="mechanics", caption="Retrieval flow",
                           prompt="labelled diagram, light background")
        return f"Body text for: {prompt.splitlines()[0][:40]}"

    backend.ask = fake_ask


def run() -> dict:
    backend.MAX_REVISIONS = 2
    call_log.clear()
    return graph.invoke(
        {
            "topic": "vector databases", "voice": "practical", "as_of": "2026-01-01",
            "outline": None, "sections": {}, "critiques": {}, "revisions": {},
            "figure": None, "figure_path": "", "document": "", "notes": [],
        }
    )


def check(label: str, condition: bool) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise SystemExit(f"failed: {label}")


print("1. clean run - critic passes everything")
install_stub(["pass", "pass", "pass"])
clean = run()
check("document produced", bool(clean["document"]))
check("no revisions triggered", clean["revisions"] == {})
check("all 3 sections drafted", len(clean["sections"]) == 3)
check("outline captured", clean["outline"] is not None)

print("2. revision run - one section fails then passes")
install_stub(["revise", "pass", "pass", "pass", "pass", "pass"])
fixed = run()
check("revision recorded", fixed["revisions"] != {})
check("every pass capped at MAX_REVISIONS",
      all(count <= 2 for count in fixed["revisions"].values()))
check("doc still complete", fixed["document"].count("## ") == 3)

print("3. stubborn section - never passes, loop must terminate")
backend.MAX_REVISIONS = 2
install_stub(["revise"] * 200)
stubborn = run()
check("stopped at the cap", all(c == 2 for c in stubborn["revisions"].values()))
check("terminated without hanging", bool(stubborn["document"]))

print("4. assembly shape")
document = stubborn["document"]
check("h1 present", document.startswith("# "))
check("sections ordered", document.index("What it fixes") < document.index("How it works"))
check("provenance footer", "Assembled" in document)

print("5. fan-out ran concurrently per section")
install_stub(["pass", "pass", "pass"])
run()
drafts = [c for c in call_log if c.startswith("You write ONE section")]
check("3 independent draft calls", len(drafts) == 3)

print("\nall checks passed")