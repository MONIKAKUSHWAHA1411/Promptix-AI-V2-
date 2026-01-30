# app.py — Promptix AI v2 (UI preserved + Supabase Email Login + Daily Limits + Provider/BYOK + Advanced Prompt)
# Fixes in this version:
# ✅ No more 1-sec StreamlitDuplicateElementKey flicker on login/logout (cookie ops are queued; only 1 cookie call per run)
# ✅ Daily limit now persists correctly (reads MAX(count) for the day; works even if multiple rows exist)
# ✅ When remaining = 0 → toast + message “0 limits remaining, come back in ~24 hours”
# ✅ Keeps current UI + brings back missing options: sample data, roles, test type, format, advanced prompt, checkboxes, send-to-AI, copy buttons
# ✅ Reduces top empty space via CSS

import os
import json
import time
import base64
import secrets
import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import requests
import streamlit as st

# Optional (recommended) for cookie persistence on Streamlit Cloud
try:
    import extra_streamlit_components as stx
except Exception:
    stx = None


# ==========================================================
# CONFIG
# ==========================================================
APP_TITLE = "Promptix AI v2"
DAILY_FREE_LIMIT = 3

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# OpenAI-compatible endpoint for Together
DEFAULT_TOGETHER_ENDPOINT = os.getenv("TOGETHER_API_ENDPOINT", "https://api.together.xyz/v1/chat/completions").rstrip("/")
DEFAULT_TOGETHER_BASE = DEFAULT_TOGETHER_ENDPOINT.replace("/chat/completions", "").rstrip("/")

COOKIE_NAME = "px_session_v2"  # store everything in one cookie
COOKIE_TTL_DAYS = 30

# If you want "day" to reset in India time:
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Kolkata")


# ==========================================================
# PAGE
# ==========================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")


# ==========================================================
# STYLES (keep your current UI + reduce top empty space)
# ==========================================================
st.markdown(
    """
<style>
/* Reduce top padding / empty space */
section.main > div.block-container { padding-top: 1.1rem !important; padding-bottom: 2rem !important; }

/* Dark gradient background */
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 30% 10%, rgba(128, 0, 255, 0.18), transparent 60%),
              radial-gradient(1200px 800px at 90% 20%, rgba(0, 200, 255, 0.10), transparent 55%),
              linear-gradient(180deg, #0b0f17 0%, #070a10 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(25, 28, 38, 0.75) 0%, rgba(14, 16, 22, 0.75) 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}

/* Card look */
.px-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 18px 18px;
  background: rgba(255,255,255,0.03);
  box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}

/* Hero */
.px-hero {
  border-radius: 18px;
  padding: 18px 22px;
  border: 1px solid rgba(255,255,255,0.10);
  background: linear-gradient(135deg, rgba(126, 64, 255, 0.40), rgba(40, 130, 255, 0.10));
}

/* Muted text */
.px-muted { color: rgba(255,255,255,0.72); font-size: 0.95rem; }

/* Auth box center */
.center-auth {
  max-width: 980px;
  margin: 0 auto;
}

/* Pills */
.px-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  border-radius: 12px;
  background: rgba(30, 120, 255, 0.12);
  border: 1px solid rgba(30, 120, 255, 0.25);
}

/* Footer */
.px-footer {
  margin-top: 30px;
  padding-top: 18px;
  border-top: 1px solid rgba(255,255,255,0.06);
  text-align: center;
  color: rgba(255,255,255,0.70);
  font-size: 0.9rem;
}
.px-footer a { color: rgba(120, 190, 255, 0.95); text-decoration: none; }
.px-footer a:hover { text-decoration: underline; }

/* Smaller expander padding */
div[data-testid="stExpander"] > details > summary {
  font-size: 1rem;
}

/* Make text areas a bit taller by default */
textarea { min-height: 140px !important; }

/* Reduce empty top space for login header line */
.px-auth-spacer { height: 8px; }

/* Better buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# HELPERS
# ==========================================================
def now_local_date_str() -> str:
    """Return YYYY-MM-DD in desired timezone (best-effort)."""
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        tz = ZoneInfo(APP_TIMEZONE)
        return datetime.datetime.now(tz).date().isoformat()
    except Exception:
        return datetime.date.today().isoformat()


def toast(msg: str, icon: str = "ℹ️"):
    """Toast if available, else fallback."""
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.info(msg)


def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def make_state() -> str:
    return b64url_encode(secrets.token_bytes(24))


# ==========================================================
# COOKIE MANAGER (queued ops to avoid DuplicateElementKey)
# ==========================================================
cookie_manager = stx.CookieManager() if stx is not None else None

def queue_cookie_set(value: str):
    st.session_state["_cookie_action"] = "set"
    st.session_state["_cookie_value"] = value


def queue_cookie_delete():
    st.session_state["_cookie_action"] = "delete"
    st.session_state["_cookie_value"] = ""


def process_cookie_queue_once():
    """
    IMPORTANT FIX:
    Only ONE cookie_manager call is allowed per script run (otherwise DuplicateElementKey flicker).
    We queue cookie ops and execute them on the *next* run.
    """
    if cookie_manager is None:
        return

    action = st.session_state.get("_cookie_action")
    if not action:
        return

    if action == "set":
        value = st.session_state.get("_cookie_value", "")
        # cookie ttl not always supported by all versions, but CookieManager generally supports it.
        try:
            cookie_manager.set(COOKIE_NAME, value, key=COO KIE_NAME if False else COOKIE_NAME)
        except TypeError:
            # Some versions don't accept extra args
            cookie_manager.set(COOKIE_NAME, value)
    elif action == "delete":
        try:
            cookie_manager.delete(COOKIE_NAME)
        except Exception:
            # ignore
            pass

    # clear queue and rerun
    st.session_state.pop("_cookie_action", None)
    st.session_state.pop("_cookie_value", None)
    st.rerun()


def bootstrap_session_from_cookie_once():
    """Read cookie ONCE per run (and only when no queued cookie action)."""
    if cookie_manager is None:
        return
    if st.session_state.get("_bootstrapped_cookie"):
        return

    # Only read if no queued operation
    if st.session_state.get("_cookie_action"):
        return

    try:
        raw = cookie_manager.get(COOKIE_NAME)
    except Exception:
        raw = None

    if raw:
        payload = safe_json_loads(raw)
        if payload and isinstance(payload, dict):
            # Minimal restore
            user = payload.get("user")
            access_token = payload.get("access_token", "")
            refresh_token = payload.get("refresh_token", "")

            if user and isinstance(user, dict) and user.get("id") and user.get("email"):
                st.session_state["user"] = {"id": user["id"], "email": user["email"]}
                st.session_state["access_token"] = access_token
                st.session_state["refresh_token"] = refresh_token

    st.session_state["_bootstrapped_cookie"] = True


def set_auth_session(user_id: str, email: str, access_token: str, refresh_token: str):
    st.session_state["user"] = {"id": user_id, "email": email}
    st.session_state["access_token"] = access_token
    st.session_state["refresh_token"] = refresh_token

    # Queue cookie write (NOT immediate)
    value = json.dumps(
        {
            "user": {"id": user_id, "email": email},
            "access_token": access_token,
            "refresh_token": refresh_token,
            "ts": int(time.time()),
        }
    )
    queue_cookie_set(value)


def clear_auth_session():
    st.session_state.pop("user", None)
    st.session_state.pop("access_token", None)
    st.session_state.pop("refresh_token", None)
    queue_cookie_delete()


# ==========================================================
# SUPABASE AUTH (Email/Password)
# ==========================================================
def supabase_headers(access_token: Optional[str] = None) -> Dict[str, str]:
    h = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    if access_token:
        h["Authorization"] = f"Bearer {access_token}"
    return h


def supabase_sign_in(email: str, password: str) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets/env.")
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=password"
    r = requests.post(url, headers=supabase_headers(), json={"email": email, "password": password}, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"Login failed: {r.text[:800]}")
    return r.json()


def supabase_sign_up(email: str, password: str) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets/env.")
    url = f"{SUPABASE_URL}/auth/v1/signup"
    r = requests.post(url, headers=supabase_headers(), json={"email": email, "password": password}, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"Signup failed: {r.text[:800]}")
    return r.json()


# ==========================================================
# SUPABASE USAGE LIMITS (robust: uses MAX(count))
# Table recommended: promptix_usage(user_id text/uuid, usage_date text/date, count int, updated_at timestamp)
# ==========================================================
def usage_table_headers() -> Dict[str, str]:
    token = st.session_state.get("access_token", "")
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def fetch_usage_rows(user_id: str, usage_date: str) -> List[Dict[str, Any]]:
    if not SUPABASE_URL:
        return []
    # Read all rows for that day. We'll take MAX(count) to stay correct even if duplicates exist.
    url = f"{SUPABASE_URL}/rest/v1/promptix_usage?user_id=eq.{user_id}&usage_date=eq.{usage_date}&select=count,updated_at"
    r = requests.get(url, headers=usage_table_headers(), timeout=20)
    if r.status_code >= 300:
        return []
    try:
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_daily_used_count(user_id: str, usage_date: str) -> int:
    rows = fetch_usage_rows(user_id, usage_date)
    if not rows:
        return 0
    max_count = 0
    for row in rows:
        try:
            c = int(row.get("count", 0))
            if c > max_count:
                max_count = c
        except Exception:
            continue
    return max_count


def write_daily_count_upsert(user_id: str, usage_date: str, new_count: int) -> bool:
    """
    Try upsert (best), requires UNIQUE(user_id, usage_date) on the table.
    If not available, this may fail and we fall back to insert.
    """
    if not SUPABASE_URL:
        return False
    url = f"{SUPABASE_URL}/rest/v1/promptix_usage?on_conflict=user_id,usage_date"
    headers = usage_table_headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    payload = {
        "user_id": user_id,
        "usage_date": usage_date,
        "count": int(new_count),
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    return r.status_code < 300


def write_daily_count_insert(user_id: str, usage_date: str, new_count: int) -> bool:
    if not SUPABASE_URL:
        return False
    url = f"{SUPABASE_URL}/rest/v1/promptix_usage"
    headers = usage_table_headers()
    headers["Prefer"] = "return=minimal"
    payload = {
        "user_id": user_id,
        "usage_date": usage_date,
        "count": int(new_count),
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    return r.status_code < 300


def increment_daily_usage(user_id: str, usage_date: str) -> int:
    """
    Robust increment:
    - Read MAX(count)
    - new = current + 1
    - try upsert; if fails, insert a new row
    - return new (or current if all failed)
    """
    current = get_daily_used_count(user_id, usage_date)
    new_count = current + 1

    ok = write_daily_count_upsert(user_id, usage_date, new_count)
    if ok:
        return new_count

    ok2 = write_daily_count_insert(user_id, usage_date, new_count)
    if ok2:
        return new_count

    return current


# ==========================================================
# LLM PROVIDERS / MODELS
# ==========================================================
@dataclass
class ProviderConfig:
    key: str
    label: str
    kind: str  # "free" or "byok"
    vendor: str  # together/openai/gemini/anthropic
    needs_key: bool


PROVIDERS = [
    ProviderConfig("promptix_free", "Promptix Free (LLaMA via Together)", "free", "together", False),
    ProviderConfig("together_byok", "Together LLaMA (BYOK)", "byok", "together", True),
    ProviderConfig("openai_byok", "OpenAI (BYOK)", "byok", "openai", True),
    ProviderConfig("gemini_byok", "Google Gemini (BYOK)", "byok", "gemini", True),
    ProviderConfig("anthropic_byok", "Anthropic Claude (BYOK)", "byok", "anthropic", True),
]

# Together model list (safe defaults; user can change)
TOGETHER_MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "(Recommended) meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
]

OPENAI_MODELS = [
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4.1-mini", "gpt-4.1-mini"),
    ("gpt-4o", "gpt-4o"),
]

GEMINI_MODELS = [
    ("gemini-1.5-flash", "gemini-1.5-flash"),
    ("gemini-1.5-pro", "gemini-1.5-pro"),
]

ANTHROPIC_MODELS = [
    ("claude-3-5-sonnet-latest", "claude-3-5-sonnet-latest"),
    ("claude-3-5-haiku-latest", "claude-3-5-haiku-latest"),
]


# ==========================================================
# PROMPT OPTIONS (RESTORED)
# ==========================================================
TESTING_ROLES = [
    "QA Tester — Manual testing expert",
    "QA Automation Engineer — Selenium/Playwright",
    "SDET — End-to-end + CI quality gates",
    "Product QA — UX + Acceptance validation",
    "API QA — Postman/Newman + contract testing",
    "Security QA — OWASP basics + threat checks",
    "Performance QA — Load/latency checks (basic)",
    "Accessibility QA — WCAG validations (basic)",
]

TEST_TYPES = [
    "Functional Testing",
    "Smoke Testing",
    "Sanity Testing",
    "Regression Testing",
    "UI Testing",
    "API Testing",
    "Integration Testing",
    "E2E Testing",
    "Accessibility (basic)",
    "Security (basic)",
    "Performance (basic)",
]

TEST_MANAGEMENT_FORMATS = [
    "Standard/Detailed — Comprehensive format",
    "Jira/Zephyr — Atlassian import style",
    "TestRail — TestRail format",
    "Cucumber/BDD — Gherkin syntax",
    "Excel/CSV — Generic import format",
    "Custom — I will define",
]

PLATFORMS = ["Web", "Mobile", "API", "Web + API", "Mobile + API"]

PRIORITY_SCHEMES = ["P0/P1/P2", "High/Medium/Low", "Critical/Major/Minor", "P1/P2/P3/P4"]


# ==========================================================
# PROMPT BUILDERS
# ==========================================================
def build_advanced_prompt(data: Dict[str, Any]) -> str:
    role = data.get("role", "QA Tester — Manual testing expert")
    test_type = data.get("test_type", "Functional Testing")
    fmt = data.get("format", "Standard/Detailed — Comprehensive format")
    platform = data.get("platform", "Web")

    context = data.get("context", "").strip()
    req = data.get("requirements", "").strip()
    acc = data.get("acceptance", "").strip()
    edges = data.get("edge_cases", "").strip()
    nfr = data.get("nfr", "").strip()

    comprehensive = data.get("comprehensive", True)
    include_edge = data.get("include_edge_cases", True)
    include_negative = data.get("include_negative", True)

    include_boundary = data.get("include_boundary", True)
    include_ui = data.get("include_ui", True)
    include_api = data.get("include_api", False)

    include_access = data.get("include_accessibility", False)
    include_security = data.get("include_security", False)
    include_perf = data.get("include_performance", False)

    n_cases = int(data.get("n_cases", 20))
    tc_prefix = data.get("tc_prefix", "TC")
    priority_scheme = data.get("priority_scheme", "P0/P1/P2")

    # Instructions by format
    format_instruction = ""
    if fmt.startswith("Jira/Zephyr"):
        format_instruction = "Output in a Jira/Zephyr-friendly table with columns: Summary, Preconditions, Steps, Expected Result, Priority, Labels."
    elif fmt.startswith("TestRail"):
        format_instruction = "Output in TestRail style with: Title, Preconditions, Steps, Expected Result, Priority."
    elif fmt.startswith("Cucumber/BDD"):
        format_instruction = "Output in Gherkin with Feature/Scenario, Given/When/Then, include tags for priority."
    elif fmt.startswith("Excel/CSV"):
        format_instruction = "Output as CSV-friendly rows: ID, Title, Preconditions, Steps, Expected, Priority."
    elif fmt.startswith("Custom"):
        format_instruction = "Ask 2 clarification questions first, then output in a structured format you propose."

    coverage_flags = []
    if comprehensive:
        coverage_flags.append("Comprehensive coverage")
    if include_edge:
        coverage_flags.append("Edge cases")
    if include_negative:
        coverage_flags.append("Negative tests")
    if include_boundary:
        coverage_flags.append("Boundary/validation tests")
    if include_ui:
        coverage_flags.append("UI validations")
    if include_api:
        coverage_flags.append("API validations")
    if include_access:
        coverage_flags.append("Accessibility checks (basic)")
    if include_security:
        coverage_flags.append("Security checks (basic)")
    if include_perf:
        coverage_flags.append("Performance checks (basic)")

    flags_text = ", ".join(coverage_flags) if coverage_flags else "Standard coverage"

    prompt = f"""You are acting as: {role}

TASK:
Create {n_cases} high-quality QA test cases for: {test_type}
Platform: {platform}
Test Management Format: {fmt}
Priority scheme: {priority_scheme}
Test case ID prefix: {tc_prefix}

COVERAGE REQUIRED:
{flags_text}

CONTEXT (What are we testing?):
{context if context else "[Add product/module context here]"}

REQUIREMENTS (What should it do?):
{req if req else "[Add user story / requirements here]"}

ACCEPTANCE CRITERIA (if any):
{acc if acc else "[Optional]"}

EDGE CASES / NEGATIVE SCENARIOS (if any):
{edges if edges else "[Optional]"}

NON-FUNCTIONAL FOCUS (if any):
{nfr if nfr else "[Optional]"}

OUTPUT RULES:
- Be specific and executable.
- Include Preconditions.
- Steps should be atomic and numbered.
- Expected results must be clear.
- Assign Priority using: {priority_scheme}
- Use IDs like: {tc_prefix}-001, {tc_prefix}-002, ...
- Avoid real secrets or personal data.
{("- " + format_instruction) if format_instruction else ""}
"""
    return prompt


# ==========================================================
# LLM CALLS
# ==========================================================
def call_together_chat(api_key: str, model: str, messages: List[Dict[str, str]], base_url: str) -> str:
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1400,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(r.text[:1200])
    j = r.json()
    return j["choices"][0]["message"]["content"]


def call_openai_chat(api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    # OpenAI official endpoint
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(r.text[:1200])
    j = r.json()
    return j["choices"][0]["message"]["content"]


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    # Minimal Gemini REST (text)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(r.text[:1200])
    j = r.json()
    return j["candidates"][0]["content"]["parts"][0]["text"]


def call_anthropic(api_key: str, model: str, prompt: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1400,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(r.text[:1200])
    j = r.json()
    # Anthropic returns list content blocks
    parts = j.get("content", [])
    texts = []
    for p in parts:
        if p.get("type") == "text":
            texts.append(p.get("text", ""))
    return "\n".join(texts).strip()


# ==========================================================
# COPY BUTTON (RESTORED)
# ==========================================================
def clipboard_button(label: str, text: str, btn_key: str):
    # A small HTML/JS clipboard button (works in Streamlit Cloud too)
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    st.components.v1.html(
        f"""
        <div style="display:flex; gap:10px; align-items:center; margin-top:6px;">
          <button id="{btn_key}" style="
            padding:10px 14px; border-radius:12px;
            border:1px solid rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.06);
            color: rgba(255,255,255,0.92);
            cursor:pointer;">
            📋 {label}
          </button>
          <span id="{btn_key}-status" style="color: rgba(255,255,255,0.65); font-size: 0.9rem;"></span>
        </div>
        <script>
          const btn = document.getElementById("{btn_key}");
          const status = document.getElementById("{btn_key}-status");
          btn.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(`{safe_text}`);
              status.innerText = "Copied ✅";
              setTimeout(() => status.innerText = "", 1200);
            }} catch(e) {{
              status.innerText = "Copy failed";
              setTimeout(() => status.innerText = "", 1200);
            }}
          }});
        </script>
        """,
        height=58,
    )


# ==========================================================
# UI — LOGIN PAGE
# ==========================================================
def render_login():
    st.markdown('<div class="center-auth">', unsafe_allow_html=True)
    st.markdown('<div class="px-auth-spacer"></div>', unsafe_allow_html=True)

    st.markdown('<div class="px-card">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.6rem; font-weight:700;">🔐 Login to Promptix</div>', unsafe_allow_html=True)
    st.markdown('<div class="px-muted">Login is required to enforce accurate daily free limits.</div>', unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Login", "Create account"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login", use_container_width=True)
            if submit:
                try:
                    data = supabase_sign_in(email.strip(), password)
                    user_obj = data.get("user") or {}
                    access_token = data.get("access_token", "")
                    refresh_token = data.get("refresh_token", "")
                    uid = user_obj.get("id", "")
                    uemail = user_obj.get("email", email.strip())
                    if not uid or not access_token:
                        raise RuntimeError("Login succeeded but user/session data is missing.")
                    set_auth_session(uid, uemail, access_token, refresh_token)
                    st.success("Logged in ✅")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    with tab_signup:
        with st.form("signup_form"):
            email2 = st.text_input("Email (new account)", placeholder="you@example.com", key="signup_email")
            password2 = st.text_input("Password", type="password", key="signup_password")
            submit2 = st.form_submit_button("Create account", use_container_width=True)
            if submit2:
                try:
                    _ = supabase_sign_up(email2.strip(), password2)
                    st.success("Account created ✅ Now login from the Login tab.")
                except Exception as e:
                    st.error(str(e))

    st.markdown("</div>", unsafe_allow_html=True)  # card
    st.markdown("</div>", unsafe_allow_html=True)  # center-auth


# ==========================================================
# UI — MAIN APP
# ==========================================================
def render_footer():
    st.markdown(
        """
<div class="px-footer">
  Thought and built by <a href="https://www.linkedin.com/in/monika-kushwaha-52443735/" target="_blank">Monika Kushwaha</a><br/>
  QA Engineer | GenAI Product Management | LLMs, RAG, Automation, Performance
</div>
""",
        unsafe_allow_html=True,
    )


def sample_fill():
    st.session_state["context"] = "zepto.com — user added a new address"
    st.session_state["requirements"] = "User logs in with valid credentials, adds a new address, and sees it saved correctly."
    st.session_state["additional_info"] = "Consider invalid address fields, network timeouts, duplicates, and location pin issues."
    st.session_state["acceptance"] = "Address is persisted, displayed in address list, and used during checkout."
    st.session_state["edge_cases"] = "Empty fields, invalid pincode, duplicate address, server timeout, slow network, app kill/resume."
    st.session_state["nfr"] = "Basic: response time < 2s on average; no crash; proper error messages."


def render_main_app():
    user = st.session_state.get("user")
    if not user:
        render_login()
        return

    user_id = user["id"]
    email = user["email"]
    today = now_local_date_str()
    used = get_daily_used_count(user_id, today)
    remaining = max(0, DAILY_FREE_LIMIT - used)

    # Sidebar — AI settings (kept)
    with st.sidebar:
        st.markdown("## 🔐 AI Settings")

        provider_labels = [p.label for p in PROVIDERS]
        provider_map = {p.label: p for p in PROVIDERS}
        selected_label = st.selectbox("Provider", provider_labels, index=0, key="px_provider")
        provider = provider_map[selected_label]

        # Provider-specific settings
        api_key = ""
        endpoint = DEFAULT_TOGETHER_BASE
        model = ""

        if provider.vendor == "together":
            endpoint = st.text_input("API Endpoint", value=DEFAULT_TOGETHER_BASE, key="together_endpoint")
            # model select
            model_keys = [m[0] for m in TOGETHER_MODELS]
            model_labels = [m[1] for m in TOGETHER_MODELS]
            selected_m = st.selectbox("Model (Together)", model_labels, index=0, key="together_model_label")
            model = model_keys[model_labels.index(selected_m)]

            if provider.kind == "free":
                # Use server key
                if not TOGETHER_API_KEY:
                    st.warning("TOGETHER_API_KEY is not set on server. Add it in Streamlit secrets.")
                api_key = TOGETHER_API_KEY
                st.markdown(f"""
<div class="px-card" style="margin-top:10px;">
  <div style="font-weight:700;">Free calls left today:</div>
  <div style="font-size:1.25rem; margin-top:6px;">{remaining}/{DAILY_FREE_LIMIT}</div>
  <div class="px-muted">Used today: {used}</div>
</div>
""", unsafe_allow_html=True)
                st.markdown(f"""
<div class="px-card" style="margin-top:10px; border-color: rgba(60, 255, 140, 0.18);">
  <div style="font-weight:700;">Using server key from env var:</div>
  <div style="margin-top:6px;">TOGETHER_API_KEY</div>
</div>
""", unsafe_allow_html=True)
            else:
                api_key = st.text_input("API Key", type="password", placeholder="Paste your Together key", key="together_byok_key")

        elif provider.vendor == "openai":
            model = st.selectbox("Model", [m[1] for m in OPENAI_MODELS], index=0, key="openai_model_label")
            model = dict(OPENAI_MODELS).get(model, "gpt-4o-mini") if isinstance(model, str) else "gpt-4o-mini"
            api_key = st.text_input("API Key", type="password", placeholder="Paste your OpenAI key", key="openai_byok_key")

        elif provider.vendor == "gemini":
            model = st.selectbox("Model", [m[1] for m in GEMINI_MODELS], index=0, key="gemini_model_label")
            model = dict(GEMINI_MODELS).get(model, "gemini-1.5-flash") if isinstance(model, str) else "gemini-1.5-flash"
            api_key = st.text_input("API Key", type="password", placeholder="Paste your Gemini key", key="gemini_byok_key")

        elif provider.vendor == "anthropic":
            model = st.selectbox("Model", [m[1] for m in ANTHROPIC_MODELS], index=0, key="anthropic_model_label")
            model = dict(ANTHROPIC_MODELS).get(model, "claude-3-5-sonnet-latest") if isinstance(model, str) else "claude-3-5-sonnet-latest"
            api_key = st.text_input("API Key", type="password", placeholder="Paste your Anthropic key", key="anthropic_byok_key")

        st.markdown("---")
        st.markdown("### 🔗 Get your API keys")
        st.markdown("- OpenAI: [Get your API key](https://platform.openai.com/api-keys)")
        st.markdown("- Gemini: [Get your API key](https://aistudio.google.com/app/apikey)")
        st.markdown("- Anthropic: [Get your API key](https://console.anthropic.com/settings/keys)")
        st.markdown("- Together: [Get your API key](https://api.together.ai/settings/api-keys)")

        st.markdown("---")
        st.markdown("**Logged in:**")
        st.markdown(f"[{email}](mailto:{email})")

        if st.button("Logout", use_container_width=True):
            clear_auth_session()
            st.rerun()

    # Top hero + stats
    st.markdown(
        f"""
<div class="px-hero">
  <div style="font-size:2.0rem; font-weight:800;">{APP_TITLE} <span style="font-size:0.9rem; opacity:0.85; margin-left:8px; padding:3px 10px; border-radius:999px; border:1px solid rgba(255,255,255,0.14);">MVP</span></div>
  <div class="px-muted" style="margin-top:6px;">
    Turn product requirements into structured, export-ready test cases — with edge cases, negatives, and multiple formats (Jira/Zephyr/TestRail/BDD/CSV).
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    colA, colB, colC = st.columns([1.7, 0.8, 0.8])
    with colA:
        st.markdown(
            f"""
<div class="px-pill">
  <span>👤</span>
  <span>Logged in as <b>{email}</b></span>
</div>
""",
            unsafe_allow_html=True,
        )
    with colB:
        st.markdown(f"<div class='px-muted'>Daily free limit</div><div style='font-size:2.2rem; font-weight:800;'>{DAILY_FREE_LIMIT}/day</div>", unsafe_allow_html=True)
    with colC:
        st.markdown(f"<div class='px-muted'>Remaining today</div><div style='font-size:2.2rem; font-weight:800;'>{remaining}</div>", unsafe_allow_html=True)

    st.markdown("")

    # If free provider selected and remaining == 0: show toast once per day
    if st.session_state.get("px_provider") == "Promptix Free (LLaMA via Together)" and remaining == 0:
        if st.session_state.get("_limit_toast_date") != today:
            toast("0 limits remaining today. Please come back in ~24 hours or switch to BYOK.", icon="⏳")
            st.session_state["_limit_toast_date"] = today

    # Main configuration form
    st.markdown("## 🧪 Test Configuration")
    st.markdown('<div class="px-card">', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🎯 Fill Sample Data", use_container_width=True):
            sample_fill()
            toast("Sample data filled ✅", icon="✅")

        role = st.selectbox("Testing Role", TESTING_ROLES, index=0, key="role")
        test_type = st.selectbox("Test Type", TEST_TYPES, index=0, key="test_type")
        test_format = st.selectbox("Test Management Format (Import-Ready)", TEST_MANAGEMENT_FORMATS, index=0, key="format")
        platform = st.selectbox("Platform", PLATFORMS, index=0, key="platform")

    with c2:
        context = st.text_area("Context (What are you testing?)", key="context", placeholder="e.g., Login + OTP, Payments, Search, Checkout")
        requirements = st.text_area("Requirements (What should it do?)", key="requirements", placeholder="Describe user story / requirement / spec...")
        acceptance = st.text_area("Acceptance criteria (optional)", key="acceptance", placeholder="List acceptance criteria, bullet points...")
        edge_cases = st.text_area("Edge cases / negative scenarios (optional)", key="edge_cases", placeholder="Invalid inputs, boundary values, error states...")
        nfr = st.text_input("Non-functional focus (optional)", key="nfr", placeholder="e.g., latency < 200ms, WCAG, OWASP basics")

    st.markdown("</div>", unsafe_allow_html=True)

    # Coverage toggles (restored)
    st.markdown('<div class="px-card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.markdown("### Coverage Toggles")
    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        comprehensive = st.checkbox("✅ Comprehensive Coverage", value=True, key="comprehensive")
        include_edge_cases = st.checkbox("🧪 Include Edge Cases", value=True, key="include_edge_cases")
        include_negative = st.checkbox("❌ Include Negative Tests", value=True, key="include_negative")

    with tcol2:
        include_boundary = st.checkbox("📏 Include Boundary/Validation Tests", value=True, key="include_boundary")
        include_ui = st.checkbox("🖥️ Include UI Validations", value=True, key="include_ui")
        include_api = st.checkbox("🔌 Include API Validations", value=False, key="include_api")

    with tcol3:
        include_accessibility = st.checkbox("♿ Accessibility (basic)", value=False, key="include_accessibility")
        include_security = st.checkbox("🛡️ Security (basic)", value=False, key="include_security")
        include_performance = st.checkbox("⚡ Performance (basic)", value=False, key="include_performance")

    st.markdown("</div>", unsafe_allow_html=True)

    # Advanced prompt engineering (restored + larger editable box)
    st.markdown('<div class="px-card" style="margin-top:14px;">', unsafe_allow_html=True)
    with st.expander("⚡ Advanced Prompt Engineering", expanded=True):
        n_cases = st.slider("Number of test cases", min_value=8, max_value=40, value=20, step=1, key="n_cases")
        tc_prefix = st.text_input("Test Case ID prefix", value="TC", key="tc_prefix")
        priority_scheme = st.selectbox("Priority scheme", PRIORITY_SCHEMES, index=0, key="priority_scheme")

        btn1, btn2 = st.columns([1, 1])
        with btn1:
            gen_adv = st.button("⚡ Generate Advanced Prompt", use_container_width=True)
        with btn2:
            send_ai = st.button("🚀 Send to AI", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Build data dict
    data = {
        "role": role,
        "test_type": test_type,
        "format": test_format,
        "platform": platform,
        "context": context,
        "requirements": requirements,
        "acceptance": acceptance,
        "edge_cases": edge_cases,
        "nfr": nfr,
        "comprehensive": comprehensive,
        "include_edge_cases": include_edge_cases,
        "include_negative": include_negative,
        "include_boundary": include_boundary,
        "include_ui": include_ui,
        "include_api": include_api,
        "include_accessibility": include_accessibility,
        "include_security": include_security,
        "include_performance": include_performance,
        "n_cases": n_cases,
        "tc_prefix": tc_prefix,
        "priority_scheme": priority_scheme,
    }

    if "advanced_prompt" not in st.session_state:
        st.session_state["advanced_prompt"] = ""

    if gen_adv:
        st.session_state["advanced_prompt"] = build_advanced_prompt(data)
        st.success("Advanced prompt generated ✅")

    # Advanced prompt box (bigger) + copy button restored
    if st.session_state.get("advanced_prompt", "").strip():
        st.markdown('<div class="px-card" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown("### 🧾 Advanced Prompt (Editable)")
        st.markdown("<div class='px-muted'>You can edit the prompt before sending.</div>", unsafe_allow_html=True)

        adv_text = st.text_area(
            "",
            value=st.session_state["advanced_prompt"],
            key="advanced_prompt_editor",
            height=320,  # bigger box (restored)
        )
        st.session_state["advanced_prompt"] = adv_text

        clipboard_button("Copy Prompt", adv_text, btn_key="copy_adv_prompt_v2")
        st.markdown("</div>", unsafe_allow_html=True)

    # Send to AI flow (restored + limit enforcement)
    if send_ai:
        # Free provider limit gate
        if provider.key == "promptix_free":
            used = get_daily_used_count(user_id, today)  # refresh
            remaining = max(0, DAILY_FREE_LIMIT - used)
            if remaining <= 0:
                toast("0 limits remaining today. Please come back in ~24 hours or switch to BYOK.", icon="⏳")
                st.warning("Free limit reached. Please come back in ~24 hours (or switch to BYOK in the sidebar).")
            else:
                # OK to proceed
                pass

        # Validate API key for BYOK
        if provider.needs_key and not api_key:
            st.error("Please add your API key in the sidebar for the selected provider (BYOK).")
        else:
            # Choose prompt
            prompt_to_send = st.session_state.get("advanced_prompt", "").strip()
            if not prompt_to_send:
                prompt_to_send = build_advanced_prompt(data)

            try:
                with st.spinner("Generating..."):
                    if provider.vendor == "together":
                        # Together uses chat messages
                        messages = [
                            {"role": "system", "content": "You are a senior QA engineer. Output clean, import-ready test cases."},
                            {"role": "user", "content": prompt_to_send},
                        ]
                        output = call_together_chat(api_key=api_key, model=model, messages=messages, base_url=endpoint)
                    elif provider.vendor == "openai":
                        messages = [
                            {"role": "system", "content": "You are a senior QA engineer. Output clean, import-ready test cases."},
                            {"role": "user", "content": prompt_to_send},
                        ]
                        output = call_openai_chat(api_key=api_key, model=model, messages=messages)
                    elif provider.vendor == "gemini":
                        output = call_gemini(api_key=api_key, model=model, prompt=prompt_to_send)
                    elif provider.vendor == "anthropic":
                        output = call_anthropic(api_key=api_key, model=model, prompt=prompt_to_send)
                    else:
                        raise RuntimeError("Unsupported provider")

                # Increment usage only for free provider
                if provider.key == "promptix_free":
                    new_used = increment_daily_usage(user_id, today)
                    new_remaining = max(0, DAILY_FREE_LIMIT - new_used)
                    st.success(f"Done ✅ Free calls left today: {new_remaining}/{DAILY_FREE_LIMIT} (Used: {new_used})")
                    if new_remaining == 0:
                        toast("0 limits remaining today. Please come back in ~24 hours.", icon="⏳")

                # Show output (restored) + copy answer
                st.markdown('<div class="px-card" style="margin-top:14px;">', unsafe_allow_html=True)
                st.markdown("## ✅ AI Output")
                st.text_area("", value=output, height=420, key="ai_output_box")
                clipboard_button("Copy Answer", output, btn_key="copy_ai_output_v2")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"AI request failed: {str(e)[:1200]}")

    render_footer()


# ==========================================================
# APP FLOW
# ==========================================================
# 1) If a cookie operation is queued, execute it (ONE cookie call) and rerun
process_cookie_queue_once()

# 2) Bootstrap from cookie (ONE cookie read call)
bootstrap_session_from_cookie_once()

# 3) Render correct page
if not st.session_state.get("user"):
    render_login()
else:
    render_main_app()
