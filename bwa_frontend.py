"""Streamlit front end for BlogGpt."""

from __future__ import annotations

import io
import os
import re
import zipfile
from datetime import date
from pathlib import Path

import streamlit as st

from bwa_backend import MAX_REVISIONS, REVIEW_THRESHOLD, graph

IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def render(markdown: str) -> None:
    """Render markdown, resolving local image paths to absolute ones.

    Streamlit will not load a relative path from its media handler, so local
    references are rewritten against the working directory before display.
    """
    rendered = markdown
    for match in IMAGE_PATTERN.finditer(markdown):
        src = match.group("src")
        if not src.startswith(("http://", "https://")):
            absolute = (Path.cwd() / src).resolve()
            if absolute.exists():
                rendered = rendered.replace(f"]({src})", f"]({absolute.as_posix()})")
    st.markdown(rendered)


def bundle(document: str, name: str) -> bytes:
    """Zip the post together with its diagram so the pair stays together."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, document.encode("utf-8"))
        for path in Path.cwd().glob("*-diagram.png"):
            archive.write(path, arcname=path.name)
    return buffer.getvalue()


def blank_state(topic: str, voice: str, as_of: str) -> dict:
    return {
        "topic": topic,
        "voice": voice,
        "as_of": as_of,
        "outline": None,
        "sections": {},
        "critiques": {},
        "revisions": {},
        "figure": None,
        "figure_path": "",
        "document": "",
        "notes": [],
    }


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="BlogGpt", layout="wide")
st.title("BlogGpt")

with st.sidebar:
    st.header("New post")

    topic = st.text_area("Topic", height=110, placeholder="e.g. why vector databases lose to plain Postgres")

    voice = st.selectbox(
        "Voice",
        ["practical, no fluff", "teaching, patient", "analytical, dense", "conversational"],
    )

    as_of = st.date_input("As of", value=date.today())

    if not any(os.getenv(key) for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY")):
        st.warning("No API key found. Copy .env.example to .env and add one.", icon="⚠️")

    generate = st.button("Write post", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"Review threshold {REVIEW_THRESHOLD:.2f} · max {MAX_REVISIONS} revision passes")

    previous = sorted(
        (p for p in Path.cwd().glob("*.md") if p.name != "README.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if previous:
        st.subheader("Drafts")
        chosen = st.selectbox(
            "Open a draft",
            previous,
            format_func=lambda p: p.stem.replace("-", " ").title(),
            label_visibility="collapsed",
        )
        if st.button("Load draft", use_container_width=True):
            st.session_state["document"] = chosen.read_text(encoding="utf-8", errors="replace")
            st.session_state["result"] = None
            st.rerun()


if "document" not in st.session_state:
    st.session_state["document"] = ""
    st.session_state["result"] = None
    st.session_state["log"] = []


if generate:
    if not topic.strip():
        st.warning("Give it a topic first.")
        st.stop()

    log: list[str] = []
    state = blank_state(topic.strip(), voice, as_of.isoformat())

    with st.status("Composing…", expanded=True) as status:
        document = ""
        for chunk in graph.stream(state, stream_mode="updates"):
            for node, update in chunk.items():
                for line in update.get("notes") or []:
                    log.append(line)
                    status.write(line)
                if node == "review" and update.get("critiques"):
                    for key, critique in update["critiques"].items():
                        log.append(f"REVIEW {key}: {critique.score:.2f} {critique.verdict}")
                if node == "assemble" and update.get("document"):
                    document = update["document"]

        status.update(label="Done", state="complete", expanded=False)

    st.session_state["result"] = state
    st.session_state["document"] = document
    st.session_state["log"] = log


document = st.session_state["document"]
result = st.session_state["result"]

if not document:
    st.info("Describe a topic and press **Write post**.")
    st.stop()

outline_tab, sections_tab, preview_tab, log_tab = st.tabs(
    ["Outline", "Sections", "Post", "Log"]
)

if result and result.get("outline"):
    outline = result["outline"]
    with outline_tab:
        st.subheader(outline.title)
        if outline.subtitle:
            st.caption(outline.subtitle)
        st.write(f"**Angle** — {outline.angle}")
        st.write(f"**Audience** — {outline.audience}")
        st.dataframe(
            [
                {
                    "key": s.key,
                    "heading": s.heading,
                    "intent": s.intent,
                    "target_words": s.target_words,
                    "passes": result["revisions"].get(s.key, 0) + 1,
                }
                for s in outline.sections
            ],
            hide_index=True,
            use_container_width=True,
        )
else:
    with outline_tab:
        st.info("Outline appears after a run in this session.")

with sections_tab:
    if result and result.get("sections"):
        for key, body in result["sections"].items():
            critique = result["critiques"].get(key)
            label = f"{key} — {critique.score:.2f} {critique.verdict}" if critique else key
            with st.expander(label):
                if critique and critique.findings:
                    st.markdown("**Findings**")
                    for finding in critique.findings:
                        st.markdown(f"- {finding}")
                    st.divider()
                st.markdown(body)
    else:
        st.info("Sections appear after a run in this session.")

with preview_tab:
    render(document)
    title = result["outline"].title if result and result.get("outline") else "post"
    stem = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "post"
    filename = f"{stem}.md"

    left, right = st.columns(2)
    left.download_button(
        "Download markdown",
        data=document.encode("utf-8"),
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
    )
    right.download_button(
        "Download with figure",
        data=bundle(document, filename),
        file_name=f"{stem}.zip",
        mime="application/zip",
        use_container_width=True,
    )

with log_tab:
    entries = st.session_state.get("log") or []
    if entries:
        st.code("\n".join(entries), language="text")
    else:
        st.caption("No run recorded in this session.")
        st.json({"max_revisions": MAX_REVISIONS, "review_threshold": REVIEW_THRESHOLD})