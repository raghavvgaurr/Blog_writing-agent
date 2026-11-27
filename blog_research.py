from __future__ import annotations

import operator
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)
class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    final: str
# -----------------------------
# 2) LLM
# -----------------------------
# Llama 3.2 1B is the default. The Streamlit app can select another installed model per run.
llm = ChatOllama(model="llama3.2:1b")


@lru_cache(maxsize=4)
def _chat_model(model_name: str):
    return ChatOllama(model=model_name)


def _model_for(config: Optional[RunnableConfig] = None):
    configurable = (config or {}).get("configurable", {})
    model_name = configurable.get("ollama_model")
    return _chat_model(model_name) if model_name else llm


def _model_name(config: Optional[RunnableConfig] = None) -> str:
    return ((config or {}).get("configurable", {}) or {}).get("ollama_model", "llama3.2:1b")


def output_path_for_title(title: str) -> Path:
    safe_title = "".join(char for char in title if char.isalnum() or char in " _-.").strip(" .")
    safe_title = "_".join(safe_title.split())
    return (Path(__file__).resolve().parent / f"{safe_title}.md").resolve()


def _emit_progress(
    config: Optional[RunnableConfig], stage: str, message: str, **details
) -> None:
    callback = ((config or {}).get("configurable", {}) or {}).get("progress_callback")
    if callback:
        callback({
            "stage": stage,
            "message": message,
            "details": details,
            "time": datetime.now().astimezone().strftime("%H:%M:%S"),
        })
# -----------------------------
# 3) Router (decide upfront)
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State, config: Optional[RunnableConfig] = None) -> dict:
    topic = state["topic"]
    configurable = (config or {}).get("configurable", {})
    research_preference = configurable.get("research_preference", "automatic")
    if research_preference == "never":
        _emit_progress(config, "router", "Web research is disabled; skipping the research router.")
        _emit_progress(config, "router_done", "Web research disabled in settings; skipping Tavily.", needs_research=False, query_count=0)
        return {"needs_research": False, "mode": "closed_book", "queries": []}

    _emit_progress(config, "router", "Asking the local model whether this topic needs fresh sources.", tool="Ollama / ChatOllama", model=_model_name(config))
    model = _model_for(config)
    decider = model.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )

    needs_research = decision.needs_research
    mode = decision.mode
    queries = decision.queries

    if research_preference == "always":
        needs_research = True
        if not queries:
            queries = [
                topic,
                f"{topic} official documentation",
                f"{topic} recent developments",
            ]
        if mode == "closed_book":
            mode = "hybrid"
    _emit_progress(
        config,
        "router_done",
        f"Research decision: {mode.replace('_', ' ')}.",
        needs_research=needs_research,
        query_count=len(queries),
    )
    return {
        "needs_research": needs_research,
        "mode": mode,
        "queries": queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"
# -----------------------------
# 4) Research (Tavily) 
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State, config: Optional[RunnableConfig] = None) -> dict:
    queries = (state.get("queries", []) or [])[:10]
    max_results = 6
    raw_results: List[dict] = []
    _emit_progress(config, "research_start", f"Starting web research with {len(queries)} search queries.", query_count=len(queries))

    for index, query in enumerate(queries, start=1):
        _emit_progress(config, "tool_call", f"Calling Tavily search {index}/{len(queries)}.", tool="TavilySearchResults", query=query, search_index=index, total_queries=len(queries))
        results = _tavily_search(query, max_results=max_results)
        raw_results.extend(results)
        _emit_progress(config, "tool_result", f"Tavily returned {len(results)} results for search {index}.", tool="TavilySearchResults", query=query, search_index=index, total_queries=len(queries), result_count=len(results))

    if not raw_results:
        _emit_progress(config, "research_done", "No search results came back; continuing with an empty evidence pack.", result_count=0)
        return {"evidence": []}

    _emit_progress(config, "synthesis", f"Organizing {len(raw_results)} search results into cited evidence.", tool="Ollama / ChatOllama", model=_model_name(config), result_count=len(raw_results))
    extractor = _model_for(config).with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    _emit_progress(config, "research_done", f"Research complete: {len(dedup)} unique sources retained.", source_count=len(dedup))
    return {"evidence": list(dedup.values())}
# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State, config: Optional[RunnableConfig] = None) -> dict:
    _emit_progress(config, "planning", "Asking the local model to build the outline.", tool="Ollama / ChatOllama", model=_model_name(config))
    planner = _model_for(config).with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    _emit_progress(config, "plan_ready", f"Outline ready with {len(plan.tasks)} sections.", section_count=len(plan.tasks), title=plan.blog_title)
    return {"plan": plan}
# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]
# -----------------------------
# 7) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict, config: Optional[RunnableConfig] = None) -> dict:
    task = Task(**payload["task"])
    _emit_progress(config, "section_start", f"Writing section {task.id}: {task.title}.", tool="Ollama / ChatOllama", model=_model_name(config), task_id=task.id, title=task.title)
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = _model_for(config).invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    _emit_progress(config, "section_done", f"Finished section {task.id}: {task.title}.", task_id=task.id, title=task.title)
    return {"sections": [(task.id, section_md)]}
# -----------------------------
# 8) Reducer (merge + save)
# -----------------------------
def reducer_node(state: State, config: Optional[RunnableConfig] = None) -> dict:
    plan = state["plan"]
    _emit_progress(config, "saving", "Combining the sections and saving the Markdown file.")

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    output_path = output_path_for_title(plan.blog_title)
    output_path.write_text(final_md, encoding="utf-8")
    _emit_progress(config, "saved", f"Blog saved to {output_path}.", output_path=str(output_path))
    


    return {"final":final_md}


# -----------------------------
# 9) Build graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app