import os
import re
from datetime import date
from typing import List, Dict, Tuple

import requests
import streamlit as st


# =========================================================
# Helpers
# =========================================================
def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return None


def clear_byok_key():
    if "user_api_key" in st.session_state:
        st.session_state.user_api_key = ""


def ss_default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val


def extract_openai_responses_text(data: dict) -> str:
    """Parse OpenAI Responses API output into plain text."""
    texts = []
    for item in (data or {}).get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    texts.append(c.get("text", ""))
    if not texts and isinstance(data, dict) and data.get("output_text"):
        texts.append(str(data.get("output_text")))
    return "\n".join([t for t in texts if t]).strip()


# =========================================================
# Product/QA differentiators (Quality Score + AC mapping + Risk tags)
# =========================================================
RISK_TAGS = [
    "Auth", "Validation", "UI", "API", "Data", "Permissions", "Network",
    "Performance", "Security", "Accessibility", "Compatibility", "Localization",
    "Payments", "Notifications", "Logging/Monitoring"
]


def extract_acceptance_criteria(req_text: str) -> List[str]:
    """Extract AC-like points from the requirements text (bullets/lines/sentences)."""
    t = (req_text or "").strip()
    if not t:
        return []

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    ac = []

    # Prefer bullet/numbered lines
    bullet_like = []
    for ln in lines:
        if re.match(r"^(\-|\*|•|\d+[\)\.]|\(\d+\))\s+", ln):
            bullet_like.append(re.sub(r"^(\-|\*|•|\d+[\)\.]|\(\d+\))\s+", "", ln).strip())

    if bullet_like:
        ac = bullet_like
    else:
        # Fallback: split by sentences if no bullets
        parts = re.split(r"(?<=[\.\?!])\s+", t)
        ac = [p.strip() for p in parts if p.strip()]

    # Keep a reasonable number
    ac = [a for a in ac if len(a) >= 8]
    return ac[:12]


def quality_score_and_gaps(context: str, requirements: str, additional: str) -> Tuple[int, List[str], List[str]]:
    """
    Returns:
      score (0-100),
      gaps (missing info checks),
      suggested_acs (ready-to-add AC suggestions)
    """
    ctx = (context or "").strip()
    req = (requirements or "").strip()
    add = (additional or "").strip()

    score = 0
    gaps = []
    suggested_acs = []

    # Presence/length signals
    if len(ctx) >= 30:
        score += 20
    else:
        gaps.append("Context is short. Add: user persona, platform (web/app), and feature scope.")
        suggested_acs.append("AC: Feature scope and platform are explicitly defined (web/app/version).")

    if len(req) >= 60:
        score += 30
    else:
        gaps.append("Requirements are short. Add acceptance criteria bullets with expected behavior.")
        suggested_acs.append("AC: Clearly list acceptance criteria with expected outcomes and validations.")

    if len(add) >= 20:
        score += 10
    else:
        gaps.append("Constraints/notes missing. Add: dependencies, assumptions, out-of-scope, error handling.")
        suggested_acs.append("AC: Error handling, dependencies, and out-of-scope items are documented.")

    # Risk area coverage heuristics
    req_low = (req + " " + add).lower()

    def has_any(keys: List[str]) -> bool:
        return any(k in req_low for k in keys)

    # Validation
    if has_any(["invalid", "validation", "mandatory", "required", "format", "min", "max", "limit"]):
        score += 10
    else:
        gaps.append("No validation rules mentioned (required fields, formats, min/max).")
        suggested_acs.append("AC: Validation rules are defined (required fields, formats, min/max, error messages).")

    # Errors/timeouts
    if has_any(["error", "fail", "timeout", "retry", "offline", "network", "exception"]):
        score += 10
    else:
        gaps.append("No error scenarios mentioned (timeouts, failures, retries).")
        suggested_acs.append("AC: Define error scenarios (timeouts, network failure, retry/backoff, user messaging).")

    # Permissions/security
    if has_any(["role", "permission", "access", "auth", "otp", "login", "session", "token"]):
        score += 10
    else:
        gaps.append("No auth/permission behavior mentioned (who can do what).")
        suggested_acs.append("AC: Define access control rules (auth required, session behavior, roles/permissions).")

    # Performance/basic NFR
    if has_any(["performance", "latency", "sla", "load", "response time", "seconds", "ms"]):
        score += 10
    else:
        gaps.append("No NFRs mentioned (performance, reliability).")
        suggested_acs.append("AC: Add NFRs (latency target, reliability expectations, rate limits if any).")

    # Cap at 100
    score = min(100, score)

    # If everything is strong, add a positive suggestion
    if score >= 85 and not gaps:
        gaps.append("Looks solid. Consider adding traceability IDs for each AC (AC-1, AC-2...) for reporting.")
        suggested_acs.append("AC: Acceptance criteria are numbered (AC-1, AC-2...) for traceability.")

    return score, gaps[:6], suggested_acs[:6]


def format_instructions(test_mgmt_format: str) -> str:
    mapping = {
        "Standard/Detailed – Comprehensive format": (
            "Return detailed test cases with: Test Case ID, Title, Priority, Preconditions, "
            "Test Data, Steps (numbered), Expected Result, Post-conditions, Risk Tags, AC Reference."
        ),
        "Jira/Zephyr – Atlassian import style": (
            "Format for Jira/Zephyr: Test Summary, Precondition, Test Steps (Step | Data | Expected), "
            "Labels/Tags, Priority, AC Reference."
        ),
        "TestRail – TestRail format": (
            "Format for TestRail: Title, Preconditions, Steps (separate), Expected Results, References, Priority, AC Reference."
        ),
        "Cucumber/BDD – Gherkin syntax": (
            "Output in Gherkin: Feature, Background, Scenarios with Given/When/Then. "
            "Add tags like @risk_auth @risk_validation and include AC reference in scenario name."
        ),
        "Excel/CSV – Generic import format": (
            "Output as CSV rows with columns: ID, Title, Priority, Preconditions, TestData, Steps, Expected, Tags, ACRef."
        ),
        "Custom – I will define": (
            "Follow the custom format described in Additional Information."
        ),
    }
    return mapping.get(test_mgmt_format, mapping["Standard/Detailed – Comprehensive format"])


def build_prompt(
    role: str,
    test_type: str,
    test_mgmt_format: str,
    context: str,
    requirements: str,
    additional: str,
    comprehensive: bool,
    edge_cases: bool,
    negative_tests: bool,
    product_mode: bool,
    advanced_notes: str,
) -> str:
    flags = []
    if comprehensive:
        flags.append("Comprehensive coverage")
    if edge_cases:
        flags.append("Include edge cases")
    if negative_tests:
        flags.append("Include negative/validation tests")
    flags_txt = ", ".join(flags) if flags else "Standard coverage"

    ac_list = extract_acceptance_criteria(requirements)
    ac_block = "\n".join([f"- AC-{i+1}: {a}" for i, a in enumerate(ac_list)]) if ac_list else "- (No explicit AC list found. Derive and number ACs from requirements.)"

    product_mode_block = ""
    if product_mode:
        product_mode_block = f"""
PRODUCT MODE (IMPORTANT)
Before test cases, output:
A) Missing acceptance criteria (bulleted, 5–10 max)
B) Risk checklist (Auth/Validation/Permissions/Error handling/NFRs)
C) Mini Test Plan (Scope, In-scope, Out-of-scope, Entry/Exit criteria, Test data strategy)
"""

    return f"""
You are Promptix AI v2 — a senior QA engineer + test architect.

TASK
Generate high-quality test cases from the given product context and requirements.

TESTER ROLE: {role}
TEST TYPE: {test_type}
OUTPUT FORMAT: {test_mgmt_format}
COVERAGE FLAGS: {flags_txt}

RISK TAGS (use 1–3 per test case from this list):
{", ".join(RISK_TAGS)}

ACCEPTANCE CRITERIA (numbered for traceability — reference these in every test case):
{ac_block}

WHAT YOU ARE TESTING (CONTEXT)
{context}

WHAT IT SHOULD DO (REQUIREMENTS / ACCEPTANCE CRITERIA DETAILS)
{requirements}

ADDITIONAL INFORMATION / CONSTRAINTS
{additional}

ADVANCED NOTES (OPTIONAL)
{advanced_notes}

{product_mode_block}

OUTPUT RULES
1) {format_instructions(test_mgmt_format)}
2) Every test case MUST include:
   - AC Reference: AC-#
   - Risk Tags: 1–3 tags from the provided list
3) Add clear preconditions + test data wherever needed.
4) Include happy path + boundary + error handling.
5) Keep steps atomic and verifiable (UI action → system reaction).
6) Do NOT invent features. If ambiguous, state assumptions explicitly.
""".strip()


# =========================================================
# API Providers
# =========================================================
PROVIDERS = {
    "Promptix Free (LLaMA via Together)": {
        "endpoint": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "key_env": "TOGETHER_API_KEY",
        "key_link": "https://api.together.xyz/settings/api-keys",
    },
    "Together LLaMA (BYOK)": {
        "endpoint": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "key_env": None,
        "key_link": "https://api.together.xyz/settings/api-keys",
    },
    "OpenAI (BYOK)": {
        "endpoint": "https://api.openai.com/v1/responses",
        "model": "gpt-4o-mini",
        "key_env": None,
        "key_link": "https://platform.openai.com/api-keys",
    },
    "Google Gemini (BYOK)": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        "model": "gemini-pro",
        "key_env": None,
        "key_link": "https://aistudio.google.com/app/apikey",
    },
    "Anthropic Claude (BYOK)": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-haiku-20240307",
        "key_env": None,
        "key_link": "https://console.anthropic.com/settings/keys",
    },
}


def call_ai(provider: str, api_endpoint: str, api_model: str, api_key: str, prompt: str) -> Tuple[str, str, object]:
    """
    Returns: (text, status, raw_debug)
    """
    # Together (chat/completions or completions)
    if provider in ["Promptix Free (LLaMA via Together)", "Together LLaMA (BYOK)"]:
        headers = {"Authorization": f"Bearer {api_key}"}

        if "/chat/completions" in api_endpoint:
            payload = {
                "model": api_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 900,
                "temperature": 0.25,
            }
        else:
            payload = {
                "model": api_model,
                "prompt": prompt,
                "max_tokens": 900,
                "temperature": 0.25,
            }

        resp = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
        data = safe_json(resp)
        status = f"Together HTTP {resp.status_code}"
        raw = data if data else resp.text

        if resp.status_code >= 400 or not data:
            return "", status, raw

        if "/chat/completions" in api_endpoint:
            text = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
        else:
            text = data.get("choices", [{}])[0].get("text", "") or ""

        return text.strip(), status, raw

    # OpenAI Responses API
    if provider == "OpenAI (BYOK)":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(
            api_endpoint,
            headers=headers,
            json={"model": api_model, "input": prompt, "store": False, "max_output_tokens": 900},
            timeout=60,
        )
        data = safe_json(resp)
        status = f"OpenAI HTTP {resp.status_code}"
        raw = data if data else resp.text

        if resp.status_code >= 400 or not data:
            return "", status, raw

        return extract_openai_responses_text(data) or "", status, raw

    # Gemini
    if provider == "Google Gemini (BYOK)":
        url = f"{api_endpoint}?key={api_key}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
        data = safe_json(resp)
        status = f"Gemini HTTP {resp.status_code}"
        raw = data if data else resp.text

        if resp.status_code >= 400 or not data:
            return "", status, raw

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return (text or "").strip(), status, raw

    # Anthropic
    if provider == "Anthropic Claude (BYOK)":
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        resp = requests.post(
            api_endpoint,
            headers=headers,
            json={"model": api_model, "max_tokens": 900, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        data = safe_json(resp)
        status = f"Anthropic HTTP {resp.status_code}"
        raw = data if data else resp.text

        if resp.status_code >= 400 or not data:
            return "", status, raw

        text = (data.get("content", [{}])[0].get("text") or "").strip()
        return text, status, raw

    return "", "Unknown provider", "No debug"


# =========================================================
# Streamlit config + styling
# =========================================================
st.set_page_config(
    page_title="Promptix AI v2",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1280px; padding-top: 1.1rem; padding-bottom: 2rem;}
      .promptix-hero {
        background: linear-gradient(135deg, rgba(79, 45, 170, 0.88), rgba(46, 25, 112, 0.88));
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.09);
        margin-bottom: 14px;
      }
      .promptix-hero h1 { margin: 0; font-size: 28px; line-height: 1.2; }
      .promptix-hero p { margin: 6px 0 0; opacity: 0.92; }
      .badge {
        display:inline-block; margin-left:10px; padding: 3px 10px;
        border-radius: 999px; font-size: 12px;
        background: rgba(0, 200, 120, 0.18);
        border: 1px solid rgba(0, 200, 120, 0.25);
      }
      .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 14px 14px 10px;
      }
      .muted { opacity: 0.82; }
      .warnbox {
        background: rgba(255, 180, 0, 0.10);
        border: 1px solid rgba(255, 180, 0, 0.22);
        border-radius: 12px;
        padding: 10px 12px;
      }
      .small { font-size: 12px; opacity: 0.88; }
      .successbar {
        background: rgba(0, 200, 120, 0.15);
        border: 1px solid rgba(0, 200, 120, 0.25);
        padding: 10px 12px;
        border-radius: 12px;
      }
      .errorbar {
        background: rgba(255, 80, 80, 0.12);
        border: 1px solid rgba(255, 80, 80, 0.22);
        padding: 10px 12px;
        border-radius: 12px;
      }
      .footer {
        margin-top: 18px;
        padding-top: 12px;
        border-top: 1px solid rgba(255,255,255,0.10);
        opacity: 0.9;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Session state defaults
# =========================================================
ss_default("generated_prompt", "")
ss_default("ai_response", "")
ss_default("api_endpoint", "")
ss_default("api_model", "")
ss_default("last_api_status", "")
ss_default("last_api_raw", "")
ss_default("inline_message", "")
ss_default("inline_is_error", False)

# Export conversions (unique functionality)
ss_default("export_csv", "")
ss_default("export_gherkin", "")
ss_default("export_jira", "")

# Free tier usage (per-session/day MVP)
ss_default("usage_day", str(date.today()))
ss_default("free_calls_used", 0)
if st.session_state.usage_day != str(date.today()):
    st.session_state.usage_day = str(date.today())
    st.session_state.free_calls_used = 0

FREE_DAILY_LIMIT = int(os.getenv("PROMPTIX_FREE_DAILY_LIMIT", "10"))

# Sample data
ss_default("sample_context", "")
ss_default("sample_requirements", "")
ss_default("sample_additional", "")

# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="promptix-hero">
      <h1>Promptix AI v2 <span class="badge">MVP</span></h1>
      <p class="muted">
        Turn product requirements into structured, export-ready test cases — with edge cases, negatives,
        traceability (AC mapping), and risk tagging.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Sidebar: AI Settings + Key links (hyperlinks only)
# =========================================================
st.sidebar.title("🔐 AI Settings")
provider = st.sidebar.selectbox("Provider", list(PROVIDERS.keys()), index=0)
defaults = PROVIDERS[provider]

# Initialize endpoint/model only if empty (doesn't overwrite user changes)
if st.session_state.api_endpoint == "" or st.session_state.api_model == "":
    st.session_state.api_endpoint = defaults["endpoint"]
    st.session_state.api_model = defaults["model"]

api_endpoint = st.sidebar.text_input("API Endpoint", value=st.session_state.api_endpoint)
api_model = st.sidebar.text_input("Model", value=st.session_state.api_model)

# Provider-specific link (no raw URL visible)
st.sidebar.markdown(f"🔗 [Get your API key]({defaults['key_link']})")

with st.sidebar.expander("Show all key links", expanded=False):
    st.sidebar.markdown(f"- **OpenAI**: [Get your API key]({PROVIDERS['OpenAI (BYOK)']['key_link']})")
    st.sidebar.markdown(f"- **Gemini**: [Get your API key]({PROVIDERS['Google Gemini (BYOK)']['key_link']})")
    st.sidebar.markdown(f"- **Anthropic**: [Get your API key]({PROVIDERS['Anthropic Claude (BYOK)']['key_link']})")
    st.sidebar.markdown(f"- **Together**: [Get your API key]({PROVIDERS['Together LLaMA (BYOK)']['key_link']})")

# Key resolution
api_key = ""
if provider == "Promptix Free (LLaMA via Together)":
    env_name = defaults["key_env"]
    api_key = os.getenv(env_name, "") or st.secrets.get(env_name, "")
    remaining = max(0, FREE_DAILY_LIMIT - st.session_state.free_calls_used)
    st.sidebar.info(f"Free calls left today (this session): {remaining}/{FREE_DAILY_LIMIT}")
    if api_key:
        st.sidebar.success(f"Using server key from env var: {env_name}")
    else:
        st.sidebar.error(f"Missing key: {env_name} (add it in Streamlit Secrets)")
else:
    st.sidebar.warning("BYOK: your key is used only for this request and cleared afterwards.")
    api_key = st.sidebar.text_input("Your API Key", type="password", key="user_api_key")

s1, s2 = st.sidebar.columns(2)
with s1:
    if st.button("💾 Save", use_container_width=True):
        st.session_state.api_endpoint = api_endpoint
        st.session_state.api_model = api_model
        st.sidebar.success("Saved.")
with s2:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.generated_prompt = ""
        st.session_state.ai_response = ""
        st.session_state.last_api_status = ""
        st.session_state.last_api_raw = ""
        st.session_state.inline_message = ""
        st.session_state.export_csv = ""
        st.session_state.export_gherkin = ""
        st.session_state.export_jira = ""
        st.sidebar.success("Cleared.")

st.sidebar.caption("Security: never hardcode or commit keys to GitHub.")

# =========================================================
# Main layout: Left inputs | Right outputs
# =========================================================
left, right = st.columns([1.10, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧪 Test Configuration")

    if st.button("🎯 Fill Sample Data", use_container_width=True):
        st.session_state.sample_context = "zepto.com — user added a new address (web + mobile)"
        st.session_state.sample_requirements = (
            "1) User logs in with valid credentials.\n"
            "2) User adds a new address with required fields and saves successfully.\n"
            "3) Saved address appears in address list and can be selected during checkout.\n"
            "4) Invalid/missing fields show inline error messages.\n"
        )
        st.session_state.sample_additional = (
            "Consider duplicates, network timeouts, address validation rules, and permission/session behavior."
        )

    role = st.selectbox(
        "Testing Role",
        [
            "QA Tester – Manual testing expert",
            "Automation Tester – Test automation specialist",
            "Performance Tester – Load & performance testing",
            "Security Tester – Security & penetration testing",
            "API Tester – API & integration testing",
            "Mobile Tester – Mobile app testing specialist",
            "Accessibility Tester – WCAG & accessibility",
            "Data Quality Tester – Data validation & ETL",
            "Custom/Other – Custom testing role",
        ],
    )

    test_type = st.selectbox(
        "Test Type",
        [
            "Functional Testing",
            "Integration Testing",
            "Regression Testing",
            "Smoke Testing",
            "Sanity Testing",
            "Exploratory Testing",
            "User Acceptance Testing",
            "End-to-End Testing",
            "Custom/Other",
        ],
    )

    test_mgmt_format = st.selectbox(
        "Test Management Format (Import-Ready)",
        [
            "Standard/Detailed – Comprehensive format",
            "Jira/Zephyr – Atlassian import style",
            "TestRail – TestRail format",
            "Cucumber/BDD – Gherkin syntax",
            "Excel/CSV – Generic import format",
            "Custom – I will define",
        ],
    )

    context = st.text_area(
        "Context (What are you testing?)",
        height=120,
        value=st.session_state.sample_context,
        max_chars=4000,
    )
    requirements = st.text_area(
        "Requirements / Acceptance Criteria",
        height=140,
        value=st.session_state.sample_requirements,
        max_chars=4000,
    )
    additional = st.text_area(
        "Additional Information / Constraints",
        height=120,
        value=st.session_state.sample_additional,
        max_chars=4000,
    )

    # --------- Product mode (differentiator) ---------
    product_mode = st.checkbox("✨ Product Mode (Missing ACs + Risk checklist + Mini Test Plan)", value=True)

    # --------- QA coverage toggles ---------
    comprehensive = st.checkbox("✅ Comprehensive Coverage", value=True)
    edge_cases = st.checkbox("🔍 Include Edge Cases", value=True)
    negative_tests = st.checkbox("❌ Include Negative Tests", value=True)

    # --------- Quality score panel (differentiator) ---------
    score, gaps, suggested_acs = quality_score_and_gaps(context, requirements, additional)
    st.markdown("#### 📌 Requirement Quality")
    cqa1, cqa2 = st.columns([1, 1])
    with cqa1:
        st.metric("Clarity & Coverage Score", f"{score}/100")
    with cqa2:
        if score >= 85:
            st.success("Strong inputs — test design will be richer.")
        elif score >= 60:
            st.warning("Decent — add a few details for better coverage.")
        else:
            st.error("Needs detail — add validations, errors, and scope.")

    if gaps:
        st.caption("Quick gaps to improve QA coverage:")
        for g in gaps:
            st.write(f"- {g}")

    if suggested_acs:
        with st.expander("Suggested Acceptance Criteria (copy/paste)", expanded=False):
            st.code("\n".join([f"- {s}" for s in suggested_acs]), language="markdown")

    st.markdown(
        """
        <div class="warnbox">
          <b>Important</b>
          <ul class="small" style="margin:6px 0 0 18px;">
            <li>Avoid real secrets or client data</li>
            <li>Review AI output before use</li>
            <li>Use as assistant — not as the only source of truth</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚡ Advanced Prompt Engineering", expanded=False):
        advanced_notes = st.text_area(
            "Add constraints, risk areas, data rules, traceability needs, etc.",
            height=110,
            placeholder="Example: map tests to ACs, include severity, include API validations, include accessibility checks...",
        )

    # Buttons
    b1, b2 = st.columns(2)

    with b1:
        if st.button("⚡ Generate Prompt", use_container_width=True):
            st.session_state.generated_prompt = build_prompt(
                role, test_type, test_mgmt_format, context, requirements, additional,
                comprehensive, edge_cases, negative_tests, product_mode, advanced_notes
            )
            st.session_state.inline_message = "Prompt generated."
            st.session_state.inline_is_error = False

    with b2:
        if st.button("🚀 Send to AI", use_container_width=True):
            st.session_state.inline_message = ""
            st.session_state.inline_is_error = False

            # Free tier limit
            if provider == "Promptix Free (LLaMA via Together)":
                if st.session_state.free_calls_used >= FREE_DAILY_LIMIT:
                    st.session_state.inline_message = "Free daily limit reached. Switch to BYOK mode."
                    st.session_state.inline_is_error = True
                    st.stop()

            prompt = st.session_state.generated_prompt.strip() or build_prompt(
                role, test_type, test_mgmt_format, context, requirements, additional,
                comprehensive, edge_cases, negative_tests, product_mode, advanced_notes
            )
            st.session_state.generated_prompt = prompt

            if not api_key:
                st.session_state.inline_message = "API key missing. Add it in Streamlit Secrets or use BYOK."
                st.session_state.inline_is_error = True
                st.stop()

            try:
                text, status, raw = call_ai(provider, api_endpoint, api_model, api_key, prompt)
                st.session_state.last_api_status = status
                st.session_state.last_api_raw = raw

                if not text.strip():
                    st.session_state.inline_message = "Request failed or returned empty text. Check AI Response → Debug."
                    st.session_state.inline_is_error = True
                    st.stop()

                st.session_state.ai_response = text.strip()

                # count free usage
                if provider == "Promptix Free (LLaMA via Together)":
                    st.session_state.free_calls_used += 1

                st.session_state.inline_message = "✅ Generated! Output is visible below and in the AI Response tab."
                st.session_state.inline_is_error = False

            finally:
                if provider != "Promptix Free (LLaMA via Together)":
                    clear_byok_key()

    # Inline message
    if st.session_state.inline_message:
        if st.session_state.inline_is_error:
            st.markdown(f'<div class="errorbar">{st.session_state.inline_message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="successbar">{st.session_state.inline_message}</div>', unsafe_allow_html=True)

    # Inline output preview (so user doesn't have to switch tabs)
    if st.session_state.ai_response.strip():
        st.markdown("### ✅ Output Preview")
        st.code(st.session_state.ai_response, language="markdown")

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    tabs = st.tabs(["🧩 Generated Prompt", "🤖 AI Response"])

    with tabs[0]:
        if st.session_state.generated_prompt.strip():
            st.code(st.session_state.generated_prompt, language="markdown")
            st.caption("Use the copy button on the code block.")
        else:
            st.info("Generate a prompt to preview it here.")

    with tabs[1]:
        if st.session_state.ai_response.strip():
            st.code(st.session_state.ai_response, language="markdown")

            # Download raw output
            st.download_button(
                "⬇️ Download output (.txt)",
                data=st.session_state.ai_response,
                file_name="promptix_testcases.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.markdown("#### 📤 Export conversions (one-click)")
            st.caption("Convert your current output into CSV / Gherkin / Jira Steps format (uses the selected provider).")

            ec1, ec2, ec3 = st.columns(3)

            def run_conversion(kind: str):
                if not st.session_state.ai_response.strip():
                    return

                # Free tier limit applies here too
                if provider == "Promptix Free (LLaMA via Together)":
                    if st.session_state.free_calls_used >= FREE_DAILY_LIMIT:
                        st.session_state.inline_message = "Free daily limit reached. Switch to BYOK mode."
                        st.session_state.inline_is_error = True
                        return

                # Key must be present
                if not api_key and provider != "Promptix Free (LLaMA via Together)":
                    st.session_state.inline_message = "Enter your BYOK key in the sidebar to convert."
                    st.session_state.inline_is_error = True
                    return

                base = st.session_state.ai_response.strip()
                if kind == "csv":
                    convert_prompt = f"""
Convert the following test cases into a clean CSV.
Columns: ID, Title, Priority, Preconditions, TestData, Steps, Expected, Tags, ACRef.
Return ONLY CSV (no markdown).

TEST CASES:
{base}
""".strip()
                elif kind == "gherkin":
                    convert_prompt = f"""
Convert the following test cases into Gherkin (Cucumber/BDD).
- Use Feature, Background (if needed), and Scenarios.
- Add tags like @risk_auth @risk_validation from risk tags found in the content.
- Include AC reference in scenario title.
Return ONLY Gherkin.

TEST CASES:
{base}
""".strip()
                else:  # jira
                    convert_prompt = f"""
Convert the following test cases into Jira/Zephyr friendly format.
For each test case output:
- Test Summary:
- Precondition:
- Test Steps table (Step | Data | Expected)
- Labels/Tags:
- Priority:
- AC Reference:
Return in plain text (no markdown tables if possible — use aligned text).

TEST CASES:
{base}
""".strip()

                text, status, raw = call_ai(provider, api_endpoint, api_model, api_key, convert_prompt)
                st.session_state.last_api_status = status
                st.session_state.last_api_raw = raw

                if not text.strip():
                    st.session_state.inline_message = "Conversion failed. Check Debug."
                    st.session_state.inline_is_error = True
                    return

                if kind == "csv":
                    st.session_state.export_csv = text.strip()
                elif kind == "gherkin":
                    st.session_state.export_gherkin = text.strip()
                else:
                    st.session_state.export_jira = text.strip()

                # count free usage
                if provider == "Promptix Free (LLaMA via Together)":
                    st.session_state.free_calls_used += 1

            with ec1:
                if st.button("Convert → CSV", use_container_width=True):
                    run_conversion("csv")
            with ec2:
                if st.button("Convert → Gherkin", use_container_width=True):
                    run_conversion("gherkin")
            with ec3:
                if st.button("Convert → Jira Steps", use_container_width=True):
                    run_conversion("jira")

            # Show conversions
            if st.session_state.export_csv:
                with st.expander("✅ CSV Export", expanded=False):
                    st.code(st.session_state.export_csv, language="text")
                    st.download_button(
                        "⬇️ Download CSV",
                        data=st.session_state.export_csv,
                        file_name="promptix_testcases.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            if st.session_state.export_gherkin:
                with st.expander("✅ Gherkin Export", expanded=False):
                    st.code(st.session_state.export_gherkin, language="gherkin")
                    st.download_button(
                        "⬇️ Download Gherkin",
                        data=st.session_state.export_gherkin,
                        file_name="promptix_testcases.feature",
                        mime="text/plain",
                        use_container_width=True,
                    )

            if st.session_state.export_jira:
                with st.expander("✅ Jira/Zephyr Export", expanded=False):
                    st.code(st.session_state.export_jira, language="text")
                    st.download_button(
                        "⬇️ Download Jira text",
                        data=st.session_state.export_jira,
                        file_name="promptix_jira_zephyr.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

        else:
            st.info("No response yet. Click Send to AI.")

        if st.session_state.last_api_status:
            st.caption(st.session_state.last_api_status)

        with st.expander("🔎 Debug (last provider response)", expanded=False):
            st.code(st.session_state.last_api_raw or "No debug yet.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Footer
# =========================================================
st.markdown(
    """
    <div class="footer">
      <div class="small">
        Thought and built by <a href="https://www.linkedin.com/in/monika-kushwaha-52443735/" target="_blank">Monika Kushwaha</a>
        — QA Engineer | GenAI Product Management | LLMs, RAG, Automation, Performance
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
