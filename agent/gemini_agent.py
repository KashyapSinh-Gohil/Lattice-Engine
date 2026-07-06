"""
AEGIS Gemini agent — natural-language grid operations copilot.

Google Gemini with FUNCTION CALLING grounded in the live pipeline outputs:
the model never invents numbers — it calls tools (rankings, what-if optimizer,
transformer watchlist, benchmarks) and composes an operator briefing from results.

Degrades gracefully: with no GEMINI_API_KEY / Vertex credentials, a deterministic
rule-based planner answers using the SAME tools, so the deployed demo never breaks.

Env:
  GEMINI_API_KEY   — AI Studio key (simplest), or
  GOOGLE_CLOUD_PROJECT + GOOGLE_GENAI_USE_VERTEXAI=true — Vertex AI mode.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

MODEL = os.environ.get("VAJRA_GEMINI_MODEL", os.environ.get("AEGIS_GEMINI_MODEL",
                                                            "gemini-2.5-flash"))

SYSTEM = """You are VAJRA·GRID Copilot, the decision assistant inside a DISCOM load-dispatch
control room. The operator is managing a heatwave peak-demand emergency.
Always ground answers in tool calls — never invent grid numbers.
Be concise and operational: lead with the recommendation, then the why (reason codes,
pain/fairness trade-offs). Flag protected feeders (hospital/water/transit) explicitly.
When asked for shed plans, call run_whatif and compare the top plans.
Units: MW for load, percent for risk. This is decision support; final authority is the operator."""

SYSTEM_AGRO = """You are VAJRA·AGRO Copilot, the decision assistant for a District Agriculture
Officer / FPO advisor managing a monsoon (kharif) season with dry-spell stress.
Always ground answers in tool calls — never invent crop numbers.
Be concise and operational: lead with the recommendation, then the why (reason codes like
DRY-SPELL-REPRO, NDVI-DECLINE, TRIGGER-HIT; yield shortfall; canal-reach fairness).
When asked to allocate scarce irrigation water or relief, call run_allocation and compare
top plans — favour under-served tail-reach villages. Flag insurance-trigger villages so
payouts start early. Units: tonnes/ha for yield, megalitres for water, percent for risk.
This is decision support; final authority is the officer."""


def build_grid_tools(store) -> dict[str, dict]:
    """store: the API's GridStore with .feeders, .transformers, .system, .whatif(target_mw)."""
    return {
        "list_top_risk_feeders": {
            "fn": lambda n=8, **_: store.feeders[:int(n)],
            "decl": {"name": "list_top_risk_feeders",
                     "description": "Top feeders by Feeder Criticality Index with reason codes",
                     "parameters": {"type": "object", "properties": {
                         "n": {"type": "integer", "description": "how many (default 8)"}}}},
        },
        "get_feeder_detail": {
            "fn": lambda feeder_id="", **_: next(
                (f for f in store.feeders if f["feeder_id"] == feeder_id
                 or feeder_id.lower() in f["name"].lower()), {"error": "not found"}),
            "decl": {"name": "get_feeder_detail", "description": "Full detail for one feeder",
                     "parameters": {"type": "object", "properties": {
                         "feeder_id": {"type": "string"}}, "required": ["feeder_id"]}},
        },
        "get_transformer_watchlist": {
            "fn": lambda n=8, **_: store.transformers[:int(n)],
            "decl": {"name": "get_transformer_watchlist",
                     "description": "Transformers ranked by 72h failure probability with SHAP reasons",
                     "parameters": {"type": "object", "properties": {
                         "n": {"type": "integer"}}}},
        },
        "run_whatif": {
            "fn": lambda target_mw=None, **_: store.whatif(
                float(target_mw or store.system.get("deficit_mw", 10) or 10)),
            "decl": {"name": "run_whatif",
                     "description": "GPU-evaluate thousands of load-shed plans for a target MW "
                                    "relief; returns top plans w/ pain+fairness. Protected feeders excluded.",
                     "parameters": {"type": "object", "properties": {
                         "target_mw": {"type": "number"}}, "required": ["target_mw"]}},
        },
        "get_system_status": {
            "fn": lambda **_: {k: v for k, v in store.system.items()
                               if not isinstance(v, list)},
            "decl": {"name": "get_system_status",
                     "description": "System load, forecast peak, supply cap, deficit, temperature",
                     "parameters": {"type": "object", "properties": {}}},
        },
        "get_benchmarks": {
            "fn": lambda **_: store.benchmarks(),
            "decl": {"name": "get_benchmarks",
                     "description": "CPU vs GPU pipeline benchmark results (the acceleration proof)",
                     "parameters": {"type": "object", "properties": {}}},
        },
    }


def build_agro_tools(store) -> dict[str, dict]:
    """store: the API's AgroStore with .villages, .triggers, .system, .allocate(budget_ml)."""
    return {
        "list_top_priority_villages": {
            "fn": lambda n=8, **_: store.villages[:int(n)],
            "decl": {"name": "list_top_priority_villages",
                     "description": "Top villages by Village Advisory Priority Index (VAPI) w/ reason codes",
                     "parameters": {"type": "object", "properties": {
                         "n": {"type": "integer", "description": "how many (default 8)"}}}},
        },
        "get_village_detail": {
            "fn": lambda village_id="", **_: next(
                (v for v in store.villages if v["village_id"] == village_id
                 or village_id.lower() in v["name"].lower()), {"error": "not found"}),
            "decl": {"name": "get_village_detail", "description": "Full detail for one village",
                     "parameters": {"type": "object", "properties": {
                         "village_id": {"type": "string"}}, "required": ["village_id"]}},
        },
        "get_insurance_triggers": {
            "fn": lambda n=10, **_: store.triggers[:int(n)],
            "decl": {"name": "get_insurance_triggers",
                     "description": "Villages that crossed a PMFBY-style insurance trigger (start payouts early)",
                     "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}},
        },
        "run_allocation": {
            "fn": lambda budget_ml=None, **_: store.allocate(
                float(budget_ml or store.system.get("water_need_total_ml", 20) * 0.35 or 20)),
            "decl": {"name": "run_allocation",
                     "description": "GPU-evaluate thousands of irrigation/relief allocation plans "
                                    "for a water budget (megalitres); returns top plans maximizing "
                                    "yield saved, favouring under-served tail-reach villages.",
                     "parameters": {"type": "object", "properties": {
                         "budget_ml": {"type": "number"}}, "required": ["budget_ml"]}},
        },
        "get_system_status": {
            "fn": lambda **_: {k: v for k, v in store.system.items()
                               if not isinstance(v, list)},
            "decl": {"name": "get_system_status",
                     "description": "Season status: villages, at-risk area, insurance triggers, "
                                    "mean yield vs normal, water need",
                     "parameters": {"type": "object", "properties": {}}},
        },
        "get_benchmarks": {
            "fn": lambda **_: store.benchmarks(),
            "decl": {"name": "get_benchmarks",
                     "description": "CPU vs GPU pipeline benchmark results (the acceleration proof)",
                     "parameters": {"type": "object", "properties": {}}},
        },
    }


# ------------------------------------------------------------------ Gemini path

def _gemini_client():
    try:
        from google import genai  # noqa: PLC0415
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return genai.Client()
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
            return genai.Client(vertexai=True,
                                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    except Exception:
        pass
    return None


def _chat_gemini(client, tools: dict, message: str, history: list, system: str) -> dict:
    from google.genai import types  # noqa: PLC0415
    decls = [types.FunctionDeclaration(**t["decl"]) for t in tools.values()]
    cfg = types.GenerateContentConfig(
        system_instruction=system, tools=[types.Tool(function_declarations=decls)],
        temperature=0.2)
    contents = []
    for h in history[-8:]:
        contents.append(types.Content(role=h["role"],
                                      parts=[types.Part.from_text(text=h["text"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

    calls_made = []
    for _ in range(5):  # tool-use loop
        resp = client.models.generate_content(model=MODEL, contents=contents, config=cfg)
        cand = resp.candidates[0]
        fcalls = [p.function_call for p in cand.content.parts
                  if getattr(p, "function_call", None)]
        if not fcalls:
            return {"reply": resp.text or "", "tool_calls": calls_made, "engine": "gemini"}
        contents.append(cand.content)
        parts = []
        for fc in fcalls:
            args = dict(fc.args or {})
            result = tools[fc.name]["fn"](**args) if fc.name in tools \
                else {"error": "unknown tool"}
            calls_made.append({"tool": fc.name, "args": args})
            parts.append(types.Part.from_function_response(
                name=fc.name, response={"result": _truncate(result)}))
        contents.append(types.Content(role="tool", parts=parts))
    return {"reply": "Tool loop limit reached.", "tool_calls": calls_made, "engine": "gemini"}


def _truncate(obj: Any, limit: int = 14000):
    s = json.dumps(obj, default=str)
    return obj if len(s) <= limit else {"truncated": s[:limit]}


# ------------------------------------------------------- deterministic fallback

def _chat_fallback_grid(tools: dict, message: str) -> dict:
    m = message.lower()
    calls = []

    def use(name, **kw):
        calls.append({"tool": name, "args": kw})
        return tools[name]["fn"](**kw)

    mw_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mw|megawatt)", m)
    if any(k in m for k in ["shed", "relief", "recover", "plan"]) or mw_match:
        sysd = use("get_system_status")
        target = float(mw_match.group(1)) if mw_match else sysd.get("deficit_mw") or 10.0
        wi = use("run_whatif", target_mw=target)
        if not wi.get("plans"):
            reply = f"No feasible shed plan found for {target} MW."
        else:
            p = wi["plans"][0]
            names = ", ".join(x["name"] + f" ({x['mw']} MW)" for x in p["feeders"][:6])
            reply = (f"**Recommended shed plan for {wi['target_mw']} MW relief** — "
                     f"evaluated {wi['plans_evaluated']:,} plans in {wi['eval_seconds']}s "
                     f"({wi['plans_per_second']:,}/s, {wi['backend']}).\n\n"
                     f"Plan A: shed {p['n_feeders']} feeders → {p['relief_mw']} MW relief, "
                     f"pain {p['pain_total']}, fairness penalty {p['fairness_penalty']}.\n"
                     f"Feeders: {names}.\n"
                     f"{wi['protected_excluded']} protected feeders (hospital/water/transit) "
                     f"were excluded. Alternative plans available in the What-If panel.")
    elif any(k in m for k in ["transformer", "dt ", "fail", "burn"]):
        txs = use("get_transformer_watchlist", n=5)
        lines = [f"- {t['transformer_id']} on {t['feeder_id']}: "
                 f"{t['p_fail_72h']*100:.0f}% 72h failure risk "
                 f"[{', '.join(t['reason_codes'])}]" for t in txs]
        reply = "**Transformer watchlist (de-load first):**\n" + "\n".join(lines)
    elif any(k in m for k in ["benchmark", "gpu", "faster", "speed"]):
        b = use("get_benchmarks")
        s = (b or {}).get("summary", {})
        reply = (f"Latest benchmarks: max speedup {s.get('max_speedup')}x across "
                 f"{len(s.get('pairs', []))} scale points. See Acceleration panel.")
    elif any(k in m for k in ["status", "load", "deficit", "situation"]):
        d = use("get_system_status")
        reply = (f"System: {d['current_mw']} MW now, forecast peak {d['forecast_peak_mw']} MW "
                 f"vs supply cap {d['supply_cap_mw']} MW → **deficit {d['deficit_mw']} MW**. "
                 f"{d['temp_now_c']}°C. {d['tx_watchlist']} feeders carry high-risk transformers.")
    else:
        f = use("list_top_risk_feeders", n=5)
        lines = [f"- #{x['rank']} {x['name']} ({x['feeder_id']}): FCI {x['fci']:.2f} "
                 f"[{', '.join(x['reason_codes'])}]" for x in f]
        reply = "**Top critical feeders right now:**\n" + "\n".join(lines) + \
                "\n\nAsk me e.g. *“give me a plan to shed 40 MW”*."
    return {"reply": reply, "tool_calls": calls, "engine": "fallback-planner"}


def _chat_fallback_agro(tools: dict, message: str) -> dict:
    m = message.lower()
    calls = []

    def use(name, **kw):
        calls.append({"tool": name, "args": kw})
        return tools[name]["fn"](**kw)

    ml_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ml|megalit|megalitre|megaliter)", m)
    if any(k in m for k in ["allocat", "irrigat", "water", "relief", "distribut"]) or ml_match:
        sysd = use("get_system_status")
        budget = float(ml_match.group(1)) if ml_match else \
            round((sysd.get("water_need_total_ml") or 60) * 0.35, 1)
        al = use("run_allocation", budget_ml=budget)
        if not al.get("plans"):
            reply = f"No feasible allocation for {budget} ML."
        else:
            p = al["plans"][0]
            names = ", ".join(f"{x['name']} ({x['reach']}, {x['water_ml']}ML)"
                              for x in p["villages"][:6])
            reply = (f"**Recommended irrigation plan for {al['budget_ml']} ML** — "
                     f"evaluated {al['plans_evaluated']:,} plans in {al['eval_seconds']}s "
                     f"({al['plans_per_second']:,}/s, {al['backend']}).\n\n"
                     f"Plan A: water {p['n_villages']} villages → **{p['yield_saved_t']} t yield "
                     f"saved**, {p['water_used_ml']} ML used, {int(p['tail_share']*100)}% to "
                     f"tail-reach villages (fairness).\n{names}.")
    elif any(k in m for k in ["insur", "trigger", "payout", "claim", "pmfby"]):
        tg = use("get_insurance_triggers", n=6)
        lines = [f"- {t['name']} ({t['village_id']}): yield {t['yield_pred']} vs normal "
                 f"{t['normal_yield']} t/ha, deficit {int(t['water_deficit']*100)}% "
                 f"[{', '.join(t['reason_codes'])}]" for t in tg]
        reply = ("**Insurance-trigger villages (start payouts early):**\n" + "\n".join(lines)) \
            if lines else "No villages crossed an insurance trigger this pass."
    elif any(k in m for k in ["benchmark", "gpu", "faster", "speed"]):
        b = use("get_benchmarks"); s = (b or {}).get("summary", {})
        reply = (f"Latest benchmarks: max speedup {s.get('max_speedup')}x. See Acceleration panel.")
    elif any(k in m for k in ["status", "season", "situation", "risk", "yield"]):
        d = use("get_system_status")
        reply = (f"Season: {d['villages']} villages, {d['at_risk_pct']}% of "
                 f"{d['total_area_ha']} ha at risk, {d['insurance_triggers']} insurance triggers. "
                 f"Mean yield forecast {d['mean_yield_pred']} vs normal {d['mean_normal_yield']} "
                 f"t/ha. Est. {d['yield_saveable_total_t']} t saveable with timely water.")
    else:
        v = use("list_top_priority_villages", n=5)
        lines = [f"- #{x['rank']} {x['name']} ({x['canal_reach']}): VAPI {x['vapi']:.2f} "
                 f"[{', '.join(x['reason_codes'])}]" for x in v]
        reply = "**Top-priority villages this week:**\n" + "\n".join(lines) + \
                "\n\nAsk me e.g. *“allocate 40 ML of water”* or *“which villages hit insurance triggers?”*"
    return {"reply": reply, "tool_calls": calls, "engine": "fallback-planner"}


def chat(store, message: str, history: list | None = None, domain: str = "grid") -> dict:
    if domain == "agro":
        tools, system, fallback = build_agro_tools(store), SYSTEM_AGRO, _chat_fallback_agro
    else:
        tools, system, fallback = build_grid_tools(store), SYSTEM, _chat_fallback_grid
    client = _gemini_client()
    if client is not None:
        try:
            return _chat_gemini(client, tools, message, history or [], system)
        except Exception as e:  # quota/key issues -> keep demo alive
            out = fallback(tools, message)
            out["note"] = f"gemini unavailable ({type(e).__name__}); used fallback planner"
            return out
    return fallback(tools, message)
