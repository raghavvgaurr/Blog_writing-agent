from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from importlib import import_module
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Load .env before importing blog_research, since its Tavily tool reads this setting.
load_dotenv()

workflow = import_module("blog_research")


st.set_page_config(
    page_title="Blog Research Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(circle at 80% 0%, #1c3150 0, #0b1220 38%, #0b1220 100%); }
      [data-testid="stHeader"] { background: rgba(11, 18, 32, .72); }
      .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
      .hero { padding: 1.5rem 1.7rem; border: 1px solid rgba(148,163,184,.22); border-radius: 22px;
              background: linear-gradient(130deg, rgba(30,64,105,.72), rgba(16,26,44,.78)); margin-bottom: 1.2rem; }
      .eyebrow { color: #7dd3fc; letter-spacing: .12em; font-size: .75rem; font-weight: 700; }
      .hero h1 { margin: .35rem 0; font-size: 2.5rem; }
      .hero p { color: #cbd5e1; margin-bottom: 0; font-size: 1.05rem; }
      div[data-testid="stMetric"] { background: rgba(30,41,59,.62); border: 1px solid rgba(148,163,184,.15);
          border-radius: 16px; padding: .8rem 1rem; }
      div[data-testid="stForm"] { background: rgba(15,23,42,.72); border: 1px solid rgba(148,163,184,.18);
          border-radius: 18px; padding: 1.1rem 1.25rem; }
      .event-time { color: #94a3b8; font-size: .78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def installed_ollama_models() -> list[str] | None:
    """Return local model tags, or None when the Ollama CLI is unavailable."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    names: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        columns = line.split()
        if columns:
            names.append(columns[0])
    return names


def model_is_installed(model_name: str, installed: list[str]) -> bool:
    return any(
        tag == model_name
        or tag == f"{model_name}:latest"
        or (":" not in model_name and tag.startswith(f"{model_name}:"))
        for tag in installed
    )


def value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def progress_fraction(event: dict[str, Any], completed_sections: int, section_count: int) -> float:
    stage = event.get("stage")
    details = event.get("details", {})
    if stage == "router":
        return 0.04
    if stage == "router_done":
        return 0.12 if details.get("needs_research") else 0.22
    if stage == "research_start":
        return 0.14
    if stage == "tool_call":
        total = max(details.get("total_queries", 1), 1)
        index = details.get("search_index", 1)
        return 0.14 + 0.27 * (index - 1) / total
    if stage == "tool_result":
        total = max(details.get("total_queries", 1), 1)
        index = details.get("search_index", 1)
        return 0.14 + 0.30 * index / total
    if stage in {"synthesis", "research_done"}:
        return 0.47
    if stage == "planning":
        return 0.52
    if stage == "plan_ready":
        return 0.60
    if stage == "section_start":
        return 0.60 + 0.32 * completed_sections / max(section_count, 1)
    if stage == "section_done":
        return 0.60 + 0.32 * completed_sections / max(section_count, 1)
    if stage == "saving":
        return 0.96
    if stage == "saved":
        return 1.0
    return 0.02


def render_activity(events: list[dict[str, Any]]) -> None:
    icons = {
        "router": "🧭", "router_done": "✅", "research_start": "🔎",
        "tool_call": "🌐", "tool_result": "📥", "synthesis": "🧩",
        "research_done": "📚", "planning": "🗺️", "plan_ready": "🧱",
        "section_start": "✍️", "section_done": "✅", "saving": "💾",
        "saved": "🎉", "error": "⚠️",
    }
    with st.container(height=360, border=True):
        for index, event in reversed(list(enumerate(events, start=1))[-18:]):
            details = event.get("details", {})
            icon = icons.get(event.get("stage"), "•")
            st.markdown(f"{icon} **Step {index} · {event.get('message', '')}**")
            st.caption(event.get("time", ""))
            if details.get("tool"):
                st.caption(f"Tool: {details['tool']} · Model: {details.get('model', '—')}")
            if details.get("query"):
                st.code(details["query"], language=None)
            elif details.get("output_path"):
                st.code(details["output_path"], language=None)
            elif details.get("result_count") is not None:
                st.caption(f"Results returned: {details['result_count']}")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">LOCAL AI · RESEARCH PIPELINE</div>
      <p>Choose a topic, watch the research and writing steps, then download your finished Markdown.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("Blog Research Studio")

installed_models = installed_ollama_models()
model_options = ["llama3.2:1b", "phi3", "qwen2.5:3b", "gemma3:1b"]

with st.sidebar:
    st.header("Run settings")
    model_name = st.selectbox(
        "Writing model",
        model_options,
        index=0,
        help="Llama 3.2 1B is the default. Pick a model already installed in Ollama.",
    )
    research_label = st.selectbox(
        "Research approach",
        ["Automatic (recommended)", "Always research", "Skip web research"],
        index=0,
        help="Automatic lets the router decide from the topic. The other choices override it.",
    )
    research_preference = {
        "Automatic (recommended)": "automatic",
        "Always research": "always",
        "Skip web research": "never",
    }[research_label]
    worker_concurrency = st.slider(
        "Section writers at once",
        min_value=1,
        max_value=3,
        value=2,
        help="Lower values reduce CPU and memory pressure. Two is a reasonable starting point on an 8 GB PC.",
    )
    st.divider()
    st.caption("Ollama runs the model locally. Web research uses Tavily when enabled.")
    if installed_models is None:
        st.warning("Could not read Ollama's local model list. Make sure Ollama is installed and running.")
    elif not model_is_installed(model_name, installed_models):
        st.warning(f"{model_name} is not installed yet.")
        st.code(f"ollama pull {model_name}", language="cmd")
    else:
        st.success(f"{model_name} is installed")
    if not os.getenv("TAVILY_API_KEY"):
        st.warning("Add TAVILY_API_KEY to your .env file to use web research.")

left, middle, right = st.columns(3)
with left:
    st.metric("Default model", "Llama 3.2 · 1B")
with middle:
    st.metric("Research", "Live steps")
with right:
    st.metric("Output folder", "Project folder")

with st.form("create_blog", clear_on_submit=False):
    topic = st.text_area(
        "What would you like to write about?",
        placeholder="Example: How small language models are changing on-device AI",
        height=105,
    )
    submitted = st.form_submit_button("Research and write", type="primary", use_container_width=True)

if submitted:
    if not topic.strip():
        st.warning("Please add a topic so I know what to research.")
    elif installed_models is not None and not model_is_installed(model_name, installed_models):
        st.error(f"{model_name} is not installed. Choose an installed model or run the command shown in the sidebar.")
    else:
        event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        run_data: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        section_count = 0
        completed_sections = 0

        def emit(event: dict[str, Any]) -> None:
            event_queue.put(event)

        def run_pipeline() -> None:
            try:
                run_data["result"] = workflow.app.invoke(
                    {
                        "topic": topic.strip(),
                        "mode": "",
                        "needs_research": False,
                        "queries": [],
                        "evidence": [],
                        "plan": None,
                        "sections": [],
                        "final": "",
                    },
                    config={
                        "max_concurrency": worker_concurrency,
                        "configurable": {
                            "ollama_model": model_name,
                            "research_preference": research_preference,
                            "progress_callback": emit,
                        }
                    },
                )
            except Exception as exc:  # noqa: BLE001 - render provider, model, and tool failures in the UI.
                run_data["error"] = exc
                emit({
                    "stage": "error",
                    "message": f"The run stopped: {type(exc).__name__}: {exc}",
                    "details": {},
                    "time": time.strftime("%H:%M:%S"),
                })

        st.subheader("Live backend activity")
        progress = st.progress(0, text="Starting the workflow…")
        activity = st.empty()
        run_thread = threading.Thread(target=run_pipeline, daemon=True)
        run_thread.start()
        current_progress = 0.02

        with st.status("Working through the blog pipeline…", expanded=True) as status:
            while run_thread.is_alive() or not event_queue.empty():
                added_event = False
                while True:
                    try:
                        event = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    events.append(event)
                    added_event = True
                    stage = event.get("stage")
                    details = event.get("details", {})
                    if stage == "plan_ready":
                        section_count = details.get("section_count", 0)
                    elif stage == "section_done":
                        completed_sections += 1
                    current_progress = max(
                        current_progress,
                        progress_fraction(event, completed_sections, section_count),
                    )
                    status.update(label=event.get("message", "Working…"), state="running")
                progress.progress(
                    min(current_progress, 1.0),
                    text=f"{len(events)} backend steps · {completed_sections}/{section_count} sections written",
                )
                if added_event:
                    with activity.container():
                        render_activity(events)
                time.sleep(0.18)

            if run_data.get("error"):
                status.update(label="The run needs attention", state="error", expanded=True)
            else:
                status.update(label="Your blog is ready", state="complete", expanded=False)
                progress.progress(1.0, text=f"Complete · {len(events)} backend steps")

        if run_data.get("error"):
            st.error(str(run_data["error"]))
            st.info(
                "Check that Ollama is running and the selected model is installed. "
                "If research was requested, check TAVILY_API_KEY in the project's .env file."
            )
        else:
            st.session_state["blog_result"] = run_data["result"]
            st.session_state["blog_model"] = model_name
            st.session_state["blog_events"] = events
            st.session_state["blog_topic"] = topic.strip()

result = st.session_state.get("blog_result")
if result:
    plan = result.get("plan")
    title = value(plan, "blog_title", "Generated blog") if plan else "Generated blog"
    markdown = result.get("final", "")
    output_path = workflow.output_path_for_title(title)

    st.divider()
    st.subheader(title)
    st.caption(f"Saved in your project folder: `{output_path}` · Model: {st.session_state.get('blog_model', model_name)}")
    st.download_button(
        "Download the Markdown file",
        data=markdown,
        file_name=output_path.name,
        mime="text/markdown",
    )

    draft_tab, outline_tab, sources_tab, activity_tab = st.tabs(
        ["Finished draft", "Outline", "Research sources", "Run timeline"]
    )
    with draft_tab:
        st.markdown(markdown)
    with outline_tab:
        tasks = value(plan, "tasks", []) if plan else []
        for task in tasks:
            with st.expander(f"Section {value(task, 'id', '')} · {value(task, 'title', '')}"):
                st.write(value(task, "goal", ""))
                for bullet in value(task, "bullets", []):
                    st.markdown(f"- {bullet}")
                st.caption(f"Target: {value(task, 'target_words', '—')} words")
    with sources_tab:
        evidence = result.get("evidence", [])
        if not evidence:
            st.info("This run did not use web sources.")
        for item in evidence:
            source_title = value(item, "title", "Source")
            url = value(item, "url", "")
            snippet = value(item, "snippet", "")
            st.markdown(f"#### [{source_title}]({url})")
            if snippet:
                st.write(snippet)
            st.caption(value(item, "published_at", None) or "Publication date not provided")
    with activity_tab:
        render_activity(st.session_state.get("blog_events", []))
