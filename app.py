# app.py — Promptix AI v2 (UI refresh only; core functionality unchanged)
# Streamlit app to generate QA test-case prompts + optionally call an LLM.
# - Default mode uses YOUR Together/LLaMA key from Streamlit Secrets (TOGETHER_API_KEY)
# - Users can switch provider + paste their own key (OpenAI / Gemini / Anthropic / Together)
#
# Footer: Thought and built by Monika Kushwaha (LinkedIn hyperlink)

import os
import re
import json
import textwrap
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Promptix AI v2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# UI skin (CSS only)
# =========================
st.markdown(
    """
<style>
/* Page width + padding */
.block-container { padding-top: 1.6rem; padding-bottom: 2.2rem; max-width: 1220px; }

/* Hero */
.px-hero{
  background: linear-gradient(135deg, rgba(124,58,237,0.38), rgba(59,130,246,0.16));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
.px-title{ font-size: 34px; font-weight: 800; letter-spacing: -0.5px; margin: 0; }
.px-sub{ opacity: 0.88; margin-top: 6px; font-size: 14px; }
.px-chip{
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  font-size: 12px; opacity: 0.92; margin-left: 10px;
}

/* Cards */
.px-card{
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 16px 16px;
  margin-bottom: 14px;
}
.px-h2{ font-size: 20px; font-weight: 750; margin: 0 0 8px 0; }
.px-muted{ opacity: 0.85; font-size: 13px; }

/* Stepper */
.px-stepper{ display:flex; gap:10px; flex-wrap: wrap; margin-top: 10px; }
.px-step{
  padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
  font-size: 12px; opacity: 0.92;
}

/* Buttons */
div.stButton > button{
  border-radius: 14px !important;
  padding: 0.68rem 1.05rem !important;
  font-weight: 700 !important;
}

/* Make code blocks feel like "preview panes" */
div[data-testid="stCodeBlock"]{
  border-radius: 14px !important;
}

/* Footer */
.px-footer{
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid rgba(255,255,255,0.08);
  opacity: 0.92;
  font-size: 12.5px;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Session state
# =========================
def init_state():
    defaults = dict(
        generated_prompt="",
        ai_response="",
        quality_score=0,
        quality_gaps=[],
        suggested_ac="",
        free_calls_left=10,
        last_error="",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# =========================
# Provider config
# =========================
PROVIDERS = [
    "Promptix Free (LLaMA via Together)",
    "User: Together",
    "User: OpenAI",
    "User: Gemini",
    "User: Anthropic",
]

API_KEY_LINKS = {
    "OpenAI": "https://platform.openai.com/api-keys",
    "Gemini": "https://aistudio.google.com/apikey",
    "Anthropic": "https://console.anthropic.com/settings/keys",
    "Together": "https://api.together.xyz/settings/api-keys",
}


def get_secret(name: str) -> str:
    # works on local env or Streamlit cloud secrets
    if name in st.secrets:
        return str(st.secrets.get(name, "")).strip()
    return os.getenv(name, "").strip()


def provider_defaults(provider: str) -> Tuple[str, str]:
    # endpoint, model
    if provider in ("Promptix Free (LLaMA via Together)", "User: Together"):
        return "https://api.together.xyz/v1/chat/completions", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    if provider == "User: OpenAI":
        return "https://api.openai.com/v1/chat/completions", "gpt-4o-mini"
    if provider == "User: Anthropic":
        return "https://api.anthropic.com/v1/messages", "claude-3-5-sonnet-latest"
    if provider == "User: Gemini":
        # We'll build the full URL from model at call-time; keep here for display.
        return "https://generativelanguage.googleapis.com/v1beta/models", "gemini-1.5-flash"
    return "", ""


# =========================
# Heuristic requirement-quality scoring
# =========================
NFR_KEYWORDS = [
    "performance", "latency", "throughput", "reliability", "availability", "security",
    "privacy", "accessibility", "a11y", "usability", "scalability", "logging", "monitoring"
]

AMBIGUOUS = ["etc", "something", "maybe", "should work", "and so on", "as needed", "like"]


def compute_quality(context: str, requirements: str, include_negatives: bool) -> Tuple[int, List[str], str]:
    ctx = (context or "").strip()
    req = (requirements or "").strip()

    score = 50
    gaps = []

    # Length / clarity
    if len(ctx) >= 25:
        score += 8
    else:
        gaps.append("Context is too short — add platform, persona, and scenario details.")
        score -= 6

    if len(req) >= 60:
        score += 10
    else:
        gaps.append("Requirements are brief — add field-level validations and expected UI/system outcomes.")
        score -= 8

    # Acceptance criteria structure
    ac_lines = [ln.strip() for ln in req.splitlines() if ln.strip()]
    numbered = sum(1 for ln in ac_lines if re.match(r"^\s*(\d+[\).\]]|\-)\s+", ln))
    if numbered >= 3:
        score += 8
    else:
        gaps.append("Add more explicit acceptance criteria (numbered list works best).")
        score -= 5

    # Platform hints
    platform_hits = 0
    for k in ["web", "mobile", "android", "ios", "api", "backend", "frontend"]:
        if k in (ctx + " " + req).lower():
            platform_hits += 1
    if platform_hits >= 2:
        score += 6
    else:
        gaps.append("Mention platform + surfaces (web/mobile/API) for stronger coverage.")
        score -= 3

    # NFR mention
    has_nfr = any(k in (ctx + " " + req).lower() for k in NFR_KEYWORDS)
    if not has_nfr:
        gaps.append("No NFRs mentioned (performance, reliability, security, accessibility).")
        score -= 6
    else:
        score += 5

    # Ambiguity penalty
    if any(a in (ctx + " " + req).lower() for a in AMBIGUOUS):
        gaps.append("Some wording is ambiguous — replace “etc/maybe” with measurable expected outcomes.")
        score -= 4

    # Negative tests flag
    if include_negatives:
        score += 3
    else:
        gaps.append("Consider enabling negative tests for stronger defect discovery.")

    # Clamp
    score = max(0, min(100, score))

    # Suggested AC template (copy/paste)
    suggested_ac = textwrap.dedent(
        """\
        AC-1: Happy path completes successfully and updates are visible immediately.
        AC-2: Required fields validate with clear inline errors (empty/invalid formats).
        AC-3: Server/network failure shows non-blocking error + retry guidance.
        AC-4: Data persists across refresh/relogin and appears in listing/history.
        AC-5: Authorization prevents access/modification for unauthorized users.
        """
    )

    return score, gaps[:6], suggested_ac


# =========================
# Prompt generation (core)
# =========================
def build_prompt(payload: Dict) -> str:
    role = payload["role"]
    test_type = payload["test_type"]
    output_format = payload["output_format"]
    context = payload["context"].strip()
    requirements = payload["requirements"].strip()

    flags = payload["coverage_flags"]
    risk_tags = payload["risk_tags"]

    # Acceptance criteria mapping (optional)
    ac_text = payload.get("acceptance_criteria_hint", "").strip()

    # Output rules
    output_rules = [
        "Return detailed test cases with: Test Case ID, Title, Priority, Preconditions, Steps, Test Data, Expected Result.",
        "Every test case MUST include: AC reference (AC-#) and 1–3 risk tags from the provided list.",
        "Include happy path + boundary + error handling. Keep steps atomic and verifiable (UI action → system reaction).",
        "Do NOT invent features. If ambiguous, state assumptions explicitly before the test cases.",
    ]
    if "Include Negative Tests" in flags:
        output_rules.insert(2, "Include negative tests and invalid inputs (missing/invalid fields, permissions, API failures).")

    if "Include NFRs (Perf/Sec/A11y)" in flags:
        output_rules.append("Add a small NFR section: performance, reliability, security, accessibility checks (as applicable).")

    prompt = f"""
You are Promptix AI v2 — a senior QA engineer + test architect.

TASK
Generate high-quality test cases from the given product context and requirements.

TESTER ROLE: {role}
TEST TYPE: {test_type}
OUTPUT FORMAT: {output_format}

COVERAGE FLAGS: {", ".join(flags) if flags else "Standard coverage"}

RISK TAGS (use 1–3 per test case from this list):
{", ".join(risk_tags)}

ACCEPTANCE CRITERIA (numbered for traceability; reference as AC-#):
{requirements if requirements else "(none provided)"}

{("SUGGESTED AC TEMPLATE (if needed):\\n" + ac_text) if ac_text else ""}

WHAT YOU ARE TESTING (CONTEXT)
{context if context else "(no context provided)"}

OUTPUT RULES
{chr(10).join([f"{i+1}) {rule}" for i, rule in enumerate(output_rules)])}

NOW GENERATE THE TEST CASES.
""".strip()

    return prompt


# =========================
# LLM Calls (core)
# =========================
def call_together(endpoint: str, model: str, api_key: str, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1400,
    }
    r = requests.post(endpoint, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def call_openai(endpoint: str, model: str, api_key: str, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1400,
    }
    r = requests.post(endpoint, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def call_anthropic(endpoint: str, model: str, api_key: str, prompt: str) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 1400,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(endpoint, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Anthropic returns list of content blocks
    return "".join([c.get("text", "") for c in data.get("content", [])]).strip()


def call_gemini(base_url: str, model: str, api_key: str, prompt: str) -> str:
    # base_url is like: https://generativelanguage.googleapis.com/v1beta/models
    url = f"{base_url}/{model}:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return json.dumps(data, indent=2)
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join([p.get("text", "") for p in parts]).strip()


def call_llm(provider: str, endpoint: str, model: str, api_key: str, prompt: str) -> str:
    if provider in ("Promptix Free (LLaMA via Together)", "User: Together"):
        return call_together(endpoint, model, api_key, prompt)
    if provider == "User: OpenAI":
        return call_openai(endpoint, model, api_key, prompt)
    if provider == "User: Anthropic":
        return call_anthropic(endpoint, model, api_key, prompt)
    if provider == "User: Gemini":
        return call_gemini(endpoint, model, api_key, prompt)
    raise ValueError("Unsupported provider")


# =========================
# Sidebar — AI Settings
# =========================
st.sidebar.markdown("## 🧠 AI Settings")

provider = st.sidebar.selectbox("Provider", PROVIDERS, index=0)

default_endpoint, default_model = provider_defaults(provider)

# Endpoint & model controls (kept, but UI looks different now)
endpoint = st.sidebar.text_input("API Endpoint", value=default_endpoint)
model = st.sidebar.text_input("Model", value=default_model)

# Key management
together_key = get_secret("TOGETHER_API_KEY")  # used for Promptix Free
user_key = ""

if provider == "Promptix Free (LLaMA via Together)":
    st.sidebar.markdown("🔒 **Using your Together key from Streamlit Secrets** (`TOGETHER_API_KEY`).")
    st.sidebar.info(f"Free calls left today (this session): **{st.session_state.free_calls_left}/10**")
    if not together_key:
        st.sidebar.error("Missing key: **TOGETHER_API_KEY** (add it in Streamlit Secrets).")
else:
    user_key = st.sidebar.text_input("Your API Key", type="password", value="")
    st.sidebar.caption("Tip: Keep keys private — don’t paste real client data into prompts.")

# API key links
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 **Get your API key**")
st.sidebar.markdown(f"- OpenAI: [{API_KEY_LINKS['OpenAI']}]({API_KEY_LINKS['OpenAI']})")
st.sidebar.markdown(f"- Gemini: [{API_KEY_LINKS['Gemini']}]({API_KEY_LINKS['Gemini']})")
st.sidebar.markdown(f"- Anthropic: [{API_KEY_LINKS['Anthropic']}]({API_KEY_LINKS['Anthropic']})")
st.sidebar.markdown(f"- Together: [{API_KEY_LINKS['Together']}]({API_KEY_LINKS['Together']})")


# =========================
# Header / Hero
# =========================
st.markdown(
    """
<div class="px-hero">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap: wrap;">
    <div>
      <h1 class="px-title">Promptix AI v2 <span class="px-chip">MVP</span></h1>
      <div class="px-sub">Turn product requirements into structured, export-ready test cases — with edge cases, negatives, traceability (AC mapping), and risk tagging.</div>
      <div class="px-stepper">
        <span class="px-step">1) Scenario Builder</span>
        <span class="px-step">2) Coverage Scoreboard</span>
        <span class="px-step">3) Prompt → AI</span>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# Sample filler
# =========================
SAMPLE = {
    "context": "zepto.com — user added a new address (web + mobile)",
    "requirements": "1) User logs in with valid credentials.\n2) User adds a new address with required fields and saves successfully.\n3) Saved address appears in address list and can be selected.\n4) Invalid/missing fields show inline error messages.",
}


def fill_sample():
    st.session_state["context_in"] = SAMPLE["context"]
    st.session_state["req_in"] = SAMPLE["requirements"]


# =========================
# Main layout (UI refresh)
# =========================
left, right = st.columns([0.58, 0.42], gap="large")

with left:
    st.markdown('<div class="px-card">', unsafe_allow_html=True)
    st.markdown('<div class="px-h2">🧩 Scenario Builder</div>', unsafe_allow_html=True)
    st.markdown('<div class="px-muted">Define what you’re testing. Keep it crisp + testable.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([0.5, 0.5], gap="small")
    with c1:
        if st.button("🎯 Fill Sample Data", use_container_width=True):
            fill_sample()
            st.rerun()

    # Use a form so inputs don’t rerun the whole app on each keystroke
    with st.form("scenario_form", clear_on_submit=False):
        role = st.selectbox(
            "Testing Role",
            ["QA Tester – Manual testing expert", "QA Engineer – Hybrid (manual + automation)", "SDET – Automation-first", "QA Lead – Strategy & coverage"],
            index=0,
        )

        test_type = st.selectbox(
            "Test Type",
            ["Functional Testing", "Regression Testing", "API Testing", "UAT / Business Testing", "Performance Smoke", "Security Smoke"],
            index=0,
        )

        output_format = st.selectbox(
            "Test Management Format (Import-Ready)",
            ["Standard/Detailed – Comprehensive format", "Jira-friendly – concise steps + expected", "TestRail – structured fields", "BDD (Gherkin) – Given/When/Then"],
            index=0,
        )

        context = st.text_area(
            "Context (What are you testing?)",
            key="context_in",
            height=90,
            placeholder="Example: zepto.com — user adds a new address (web + mobile)…",
        )

        requirements = st.text_area(
            "Requirements / Acceptance Criteria",
            key="req_in",
            height=140,
            placeholder="Numbered acceptance criteria works best (AC-1, AC-2...)",
        )

        st.markdown("")

        # Coverage flags (kept — same functionality)
        f1, f2, f3 = st.columns([0.34, 0.33, 0.33], gap="small")
        with f1:
            include_neg = st.checkbox("Include Negative Tests", value=True)
        with f2:
            include_edges = st.checkbox("Include Edge/Boundary", value=True)
        with f3:
            include_nfr = st.checkbox("Include NFRs (Perf/Sec/A11y)", value=False)

        with st.expander("⚡ Advanced Prompt Controls", expanded=False):
            risk_tags_raw = st.text_input(
                "Risk Tags (comma-separated)",
                value="Auth, Validation, UI, API, Data, Permissions, Network, Performance, Security, Accessibility",
            )
            risk_tags = [t.strip() for t in risk_tags_raw.split(",") if t.strip()]

            extra_instructions = st.text_area(
                "Extra Instructions (optional)",
                height=80,
                placeholder="Example: prioritize mobile flows; include offline/network loss; focus on address validations…",
            )

        gen = st.form_submit_button("⚡ Generate Prompt", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Coverage Scoreboard (computed whenever we generate)
    st.markdown('<div class="px-card">', unsafe_allow_html=True)
    st.markdown('<div class="px-h2">📌 Coverage Scoreboard</div>', unsafe_allow_html=True)

    # Compute quality live if something exists (or after generate)
    score, gaps, suggested = compute_quality(context or "", requirements or "", include_neg)

    # store latest so UI stays consistent after rerun
    st.session_state.quality_score = score
    st.session_state.quality_gaps = gaps
    st.session_state.suggested_ac = suggested

    qc1, qc2 = st.columns([0.35, 0.65], gap="large")
    with qc1:
        st.metric("Clarity & Coverage", f"{score}/100")
    with qc2:
        if score >= 85:
            st.success("Strong inputs — test design will be richer.")
        elif score >= 70:
            st.warning("Decent inputs — add a bit more detail to boost coverage.")
        else:
            st.error("Inputs are thin — add explicit validations + expected outcomes.")

    if gaps:
        st.markdown("**Quick gaps to improve QA coverage:**")
        for g in gaps:
            st.markdown(f"- {g}")

    with st.expander("Suggested Acceptance Criteria (copy/paste)", expanded=False):
        st.code(suggested, language="text")

    st.info("**Important**\n\n- Avoid real secrets or client data\n- Review AI output before use\n- Use as assistant — not as the only source of truth")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="px-card">', unsafe_allow_html=True)
    st.markdown('<div class="px-h2">🧪 Output Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="px-muted">Generate the prompt first, then send to AI.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧩 Generated Prompt", "🤖 AI Response"])

    # Build payload and prompt when user hits Generate
    coverage_flags = []
    if include_edges:
        coverage_flags.append("Include Edge/Boundary")
    if include_neg:
        coverage_flags.append("Include Negative Tests")
    if include_nfr:
        coverage_flags.append("Include NFRs (Perf/Sec/A11y)")

    payload = {
        "role": role,
        "test_type": test_type,
        "output_format": output_format,
        "context": context or "",
        "requirements": requirements or "",
        "coverage_flags": coverage_flags,
        "risk_tags": risk_tags if "risk_tags" in locals() else ["Auth", "Validation", "UI", "API", "Data"],
        "acceptance_criteria_hint": st.session_state.suggested_ac if (requirements or "").strip() == "" else "",
    }

    if gen:
        prompt = build_prompt(payload)
        if "extra_instructions" in locals() and extra_instructions.strip():
            prompt += "\n\nADDITIONAL INSTRUCTIONS\n" + extra_instructions.strip()

        st.session_state.generated_prompt = prompt
        st.session_state.ai_response = ""
        st.success("Prompt generated.")

    with tab1:
        if st.session_state.generated_prompt:
            st.code(st.session_state.generated_prompt, language="text")
        else:
            st.info("Generate a prompt to preview it here.")

    with tab2:
        if st.session_state.ai_response:
            st.write(st.session_state.ai_response)
        else:
            st.info("Send to AI to view the response here.")

    st.markdown("---")

    # Actions row (core: generate + send)
    a1, a2 = st.columns([0.5, 0.5], gap="small")

    with a1:
        if st.button("⚡ Generate Prompt (quick)", use_container_width=True):
            prompt = build_prompt(payload)
            if "extra_instructions" in locals() and extra_instructions.strip():
                prompt += "\n\nADDITIONAL INSTRUCTIONS\n" + extra_instructions.strip()
            st.session_state.generated_prompt = prompt
            st.session_state.ai_response = ""
            st.toast("Prompt updated.", icon="✅")
            st.rerun()

    with a2:
        if st.button("🚀 Send to AI", use_container_width=True):
            # If no prompt, auto-generate first (no new functionality; just smoother UX)
            if not st.session_state.generated_prompt:
                prompt = build_prompt(payload)
                if "extra_instructions" in locals() and extra_instructions.strip():
                    prompt += "\n\nADDITIONAL INSTRUCTIONS\n" + extra_instructions.strip()
                st.session_state.generated_prompt = prompt

            # Determine which key to use
            active_key = together_key if provider == "Promptix Free (LLaMA via Together)" else user_key

            if provider == "Promptix Free (LLaMA via Together)":
                if not together_key:
                    st.error("Missing **TOGETHER_API_KEY**. Add it in Streamlit Secrets.")
                elif st.session_state.free_calls_left <= 0:
                    st.error("Free calls exhausted for this session (10/10).")
                else:
                    try:
                        with st.spinner("Calling AI…"):
                            st.session_state.ai_response = call_llm(
                                provider=provider,
                                endpoint=endpoint,
                                model=model,
                                api_key=active_key,
                                prompt=st.session_state.generated_prompt,
                            )
                        st.session_state.free_calls_left -= 1
                        st.success("AI response generated.")
                    except Exception as e:
                        st.session_state.last_error = str(e)
                        st.error(f"AI call failed: {e}")
            else:
                if not (user_key or "").strip():
                    st.error("Please paste your API key in the sidebar for the selected provider.")
                else:
                    try:
                        with st.spinner("Calling AI…"):
                            st.session_state.ai_response = call_llm(
                                provider=provider,
                                endpoint=endpoint,
                                model=model,
                                api_key=active_key,
                                prompt=st.session_state.generated_prompt,
                            )
                        st.success("AI response generated.")
                    except Exception as e:
                        st.session_state.last_error = str(e)
                        st.error(f"AI call failed: {e}")

            st.rerun()

    if st.session_state.last_error:
        with st.expander("Last error (debug)", expanded=False):
            st.code(st.session_state.last_error, language="text")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Footer (as requested)
# =========================
st.markdown(
    """
<div class="px-footer">
  Thought and built by
  <a href="https://www.linkedin.com/in/monika-kushwaha-52443735/" target="_blank" rel="noopener noreferrer">
    Monika Kushwaha
  </a>
  — QA Engineer | GenAI Product Management | LLMs, RAG, Automation, Performance
</div>
""",
    unsafe_allow_html=True,
)
