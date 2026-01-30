# app.py — Promptix AI v2 (FULL)
# ✅ Keeps: Current UI feel + Supabase Email Login (reload-safe)
# ✅ Restores missing options + fixes Together model error handling + copy buttons
# ✅ Adds: Platform dropdown, richer Test Type + Format options, API Endpoint display
# ✅ Fix: default Together model -> Meta-Llama 3.1 8B Turbo (more commonly accessible)

import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Tuple

import requests
import streamlit as st
import extra_streamlit_components as stx

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Promptix AI v2", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

# Server key used ONLY for "Promptix Free"
TOGETHER_API_KEY_SERVER = st.secrets.get("TOGETHER_API_KEY", "").strip()

APP_TIMEZONE = st.secrets.get("APP_TIMEZONE", "Asia/Kolkata")
DAILY_FREE_LIMIT = int(st.secrets.get("DAILY_FREE_LIMIT", 3))

# OpenAI-compatible endpoints
TOGETHER_BASE_URL_DEFAULT = st.secrets.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1").strip()
OPENAI_BASE_URL_DEFAULT = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

# Sensible defaults
# NOTE: Your screenshot error indicates the 70B name wasn't accessible. 8B Turbo is safer default.
TOGETHER_MODEL_SAFE_DEFAULT = st.secrets.get(
    "TOGETHER_MODEL_SAFE_DEFAULT", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
).strip()

TOGETHER_MODEL_ALT_70B = st.secrets.get(
    "TOGETHER_MODEL_ALT_70B", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
).strip()

OPENAI_MODEL_DEFAULT = st.secrets.get("OPENAI_MODEL_DEFAULT", "gpt-4o-mini").strip()
GEMINI_MODEL_DEFAULT = st.secrets.get("GEMINI_MODEL_DEFAULT", "gemini-1.5-flash").strip()
ANTHROPIC_MODEL_DEFAULT = st.secrets.get("ANTHROPIC_MODEL_DEFAULT", "claude-3-5-sonnet-latest").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_ANON_KEY in Streamlit Secrets.")
    st.stop()

cookie_manager = stx.CookieManager()

# =========================================================
# UI STYLES (KEEP CURRENT FEEL)
# =========================================================
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 700px at 20% 10%, rgba(255,0,90,0.10), transparent 60%),
              radial-gradient(1000px 600px at 80% 20%, rgba(0,180,255,0.10), transparent 60%),
              linear-gradient(180deg, #05070c 0%, #070a12 55%, #05070c 100%);
}
header {visibility: hidden;}
footer {visibility: hidden;}

.center-auth{
  width: 100%;
  display:flex;
  justify-content:center;
  margin-top: 52px;
}
.px-card{
  width:min(920px, 94vw);
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 26px 26px 18px 26px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.55);
}
.px-h2{ font-size: 26px; font-weight: 700; margin-bottom: 6px; }
.px-muted{ opacity: 0.75; margin-bottom: 16px; }

.hero{
  width:100%;
  background: linear-gradient(90deg, rgba(99,102,241,0.55), rgba(147,51,234,0.40));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 20px 22px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.45);
}
.badge{
  display:inline-block;
  font-size: 12px;
  padding: 2px 10px;
  margin-left: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.18);
  vertical-align: middle;
}
.subtle{
  opacity:0.80;
  margin-top: 6px;
}

.callout{
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  padding: 14px 16px;
  background: rgba(255, 196, 0, 0.10);
}

.px-footer{
  margin-top: 34px;
  text-align:center;
  opacity:0.75;
  font-size: 13px;
}
.px-footer a{ color: #7dd3fc; text-decoration:none; }
.px-footer a:hover{ text-decoration:underline; }

.px-links a { color:#7dd3fc !important; text-decoration:none; }
.px-links a:hover { text-decoration:underline; }

.kpi-box{
  border-radius: 14px;
  padding: 12px 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
}
.kpi-green{
  border-radius: 14px;
  padding: 12px 14px;
  border: 1px solid rgba(34,197,94,0.30);
  background: rgba(34,197,94,0.12);
}

/* Improve text area look */
textarea {
  border-radius: 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# TIME HELPERS
# =========================================================
def today_local_iso() -> str:
    tz = ZoneInfo(APP_TIMEZONE)
    return datetime.now(tz).date().isoformat()

def _safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return None

def _sb_headers(access_token: Optional[str] = None) -> dict:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token or SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }

# =========================================================
# SUPABASE AUTH (NO CRASH)
# =========================================================
def supabase_sign_in(email: str, password: str) -> Tuple[Optional[dict], Optional[str]]:
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=password"
    try:
        r = requests.post(url, headers=_sb_headers(None), json={"email": email, "password": password}, timeout=20)
    except Exception as e:
        return None, f"Network error: {e}"
    if r.status_code != 200:
        j = _safe_json(r)
        msg = (j.get("error_description") or j.get("message") or j.get("msg")) if isinstance(j, dict) else None
        return None, f"{r.status_code}: {msg or r.text.strip()}"
    return r.json(), None

def supabase_sign_up(email: str, password: str) -> Tuple[Optional[dict], Optional[str]]:
    url = f"{SUPABASE_URL}/auth/v1/signup"
    try:
        r = requests.post(url, headers=_sb_headers(None), json={"email": email, "password": password}, timeout=20)
    except Exception as e:
        return None, f"Network error: {e}"
    if r.status_code not in (200, 201):
        j = _safe_json(r)
        msg = (j.get("error_description") or j.get("message") or j.get("msg")) if isinstance(j, dict) else None
        return None, f"{r.status_code}: {msg or r.text.strip()}"
    return r.json(), None

def supabase_get_user(access_token: str) -> Optional[dict]:
    url = f"{SUPABASE_URL}/auth/v1/user"
    try:
        r = requests.get(url, headers=_sb_headers(access_token), timeout=15)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    return _safe_json(r)

def supabase_refresh(refresh_token: str) -> Tuple[Optional[dict], Optional[str]]:
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token"
    try:
        r = requests.post(url, headers=_sb_headers(None), json={"refresh_token": refresh_token}, timeout=20)
    except Exception as e:
        return None, f"Network error: {e}"
    if r.status_code != 200:
        j = _safe_json(r)
        msg = (j.get("error_description") or j.get("message") or j.get("msg")) if isinstance(j, dict) else None
        return None, f"{r.status_code}: {msg or r.text.strip()}"
    return r.json(), None

# =========================================================
# COOKIE SESSION
# =========================================================
COOKIE_KEYS = {
    "user_id": "px_user_id",
    "user_email": "px_user_email",
    "access_token": "px_access_token",
    "refresh_token": "px_refresh_token",
}

def save_session_to_cookie(user_id: str, email: str, access_token: str, refresh_token: str):
    max_age = 60 * 60 * 24 * 30
    cookie_manager.set(COOKIE_KEYS["user_id"], user_id, max_age=max_age)
    cookie_manager.set(COOKIE_KEYS["user_email"], email, max_age=max_age)
    cookie_manager.set(COOKIE_KEYS["access_token"], access_token, max_age=max_age)
    cookie_manager.set(COOKIE_KEYS["refresh_token"], refresh_token, max_age=max_age)

def clear_cookie_session():
    for k in COOKIE_KEYS.values():
        cookie_manager.delete(k)

def apply_auth_session(user_obj: dict, access_token: str, refresh_token: str):
    user_id = user_obj.get("id", "")
    email = user_obj.get("email", "")
    st.session_state.user = {"id": user_id, "email": email}
    st.session_state.access_token = access_token
    st.session_state.refresh_token = refresh_token
    save_session_to_cookie(user_id, email, access_token, refresh_token)

def try_bootstrap_session_from_cookie():
    if st.session_state.get("bootstrapped", False):
        return
    st.session_state.bootstrapped = True

    cookies = cookie_manager.get_all() or {}
    access_token = cookies.get(COOKIE_KEYS["access_token"])
    refresh_token = cookies.get(COOKIE_KEYS["refresh_token"])
    user_id = cookies.get(COOKIE_KEYS["user_id"])
    user_email = cookies.get(COOKIE_KEYS["user_email"])

    if not access_token or not refresh_token or not user_id:
        return

    user = supabase_get_user(access_token)
    if user and user.get("id") == user_id:
        st.session_state.user = {"id": user_id, "email": user.get("email", user_email or "")}
        st.session_state.access_token = access_token
        st.session_state.refresh_token = refresh_token
        return

    data, err = supabase_refresh(refresh_token)
    if err or not data:
        clear_cookie_session()
        return

    new_access = data.get("access_token", "")
    new_refresh = data.get("refresh_token", refresh_token)
    user_obj = data.get("user") or supabase_get_user(new_access) or {}

    if user_obj.get("id"):
        apply_auth_session(user_obj, new_access, new_refresh)

def is_authed() -> bool:
    return bool(st.session_state.get("user", {}).get("id")) and bool(st.session_state.get("access_token"))

def do_logout():
    st.session_state.pop("user", None)
    st.session_state.pop("access_token", None)
    st.session_state.pop("refresh_token", None)
    clear_cookie_session()
    st.rerun()

# =========================================================
# SUPABASE USAGE (PERSISTENT) — Promptix Free only
# =========================================================
def get_daily_usage(user_id: str, access_token: str, usage_date: str) -> int:
    url = f"{SUPABASE_URL}/rest/v1/promptix_usage?user_id=eq.{user_id}&usage_date=eq.{usage_date}&select=count"
    try:
        r = requests.get(url, headers=_sb_headers(access_token), timeout=15)
    except Exception:
        return 0
    if r.status_code != 200:
        return 0
    j = _safe_json(r)
    if isinstance(j, list) and j and isinstance(j[0], dict):
        return int(j[0].get("count", 0) or 0)
    return 0

def set_daily_usage(user_id: str, access_token: str, usage_date: str, new_count: int) -> bool:
    url = f"{SUPABASE_URL}/rest/v1/promptix_usage"
    headers = _sb_headers(access_token)
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    payload = {
        "user_id": user_id,
        "usage_date": usage_date,
        "count": int(new_count),
        "updated_at": datetime.utcnow().isoformat(),
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
    except Exception:
        return False
    return r.status_code in (201, 204)

def increment_daily_usage(user_id: str, access_token: str, usage_date: str) -> int:
    current = get_daily_usage(user_id, access_token, usage_date)
    new_count = current + 1
    ok = set_daily_usage(user_id, access_token, usage_date, new_count)
    return new_count if ok else current

# =========================================================
# TOGETHER MODEL HELPERS (for the "Unable to access model" error)
# =========================================================
TOGETHER_SUGGESTED_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
]

def together_list_models(api_key: str) -> Optional[set]:
    """Best-effort model list (no hard failure if it can't fetch)."""
    if not api_key:
        return None
    try:
        r = requests.get(
            TOGETHER_BASE_URL_DEFAULT.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        j = r.json()
        data = j.get("data", [])
        names = set()
        for m in data:
            if isinstance(m, dict) and m.get("id"):
                names.add(m["id"])
        return names or None
    except Exception:
        return None

# =========================================================
# LLM CALLS
# =========================================================
def call_openai_compatible(base_url: str, api_key: str, model: str, prompt: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a senior QA engineer. Follow instructions exactly and output in requested format."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.35,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        j = _safe_json(r)
        msg = None
        if isinstance(j, dict):
            msg = (j.get("error") or {}).get("message") or j.get("message")
        raise RuntimeError(msg or r.text)
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

def call_gemini(api_key: str, model: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.35},
    }
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=90)
    if r.status_code != 200:
        j = _safe_json(r)
        msg = j.get("error", {}).get("message") if isinstance(j, dict) else None
        raise RuntimeError(msg or r.text)
    j = r.json()
    candidates = j.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()

def call_anthropic(api_key: str, model: str, prompt: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1800,
        "temperature": 0.35,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        j = _safe_json(r)
        msg = j.get("error", {}).get("message") if isinstance(j, dict) else None
        raise RuntimeError(msg or r.text)
    j = r.json()
    content = j.get("content", [])
    if not content:
        return ""
    return content[0].get("text", "").strip()

# =========================================================
# PROMPT ENGINEERING (RESTORED + EXPANDED OPTIONS)
# =========================================================
TESTING_ROLES = [
    "QA Tester — Manual testing expert",
    "QA Engineer — Functional + Regression",
    "QA Automation Engineer — Selenium/Playwright",
    "SDET — Automation + Framework design",
    "QA Lead — Strategy + Risk-based testing",
    "Product QA — UX + Acceptance validation",
    "API QA — Contract + Integration testing",
    "Performance QA — Load/Stress basics",
    "Security QA — OWASP basics",
    "Accessibility QA — WCAG basics",
]

# This "Test Type" dropdown is the classic QA type (functional/smoke/etc.)
TEST_TYPES = [
    "Functional Testing",
    "Smoke Testing",
    "Sanity Testing",
    "Regression Testing",
    "UI Testing",
    "API Testing",
    "Integration Testing",
    "End-to-End (E2E) Testing",
    "Accessibility Testing",
    "Security Testing (basic)",
    "Performance Testing (basic)",
]

PLATFORMS = ["Web", "Mobile", "API", "Web + API", "Mobile + API"]

# Match your screenshot style (Jira/Zephyr, Excel/CSV, Custom)
TEST_MGMT_FORMATS = [
    "Standard/Detailed — Comprehensive format",
    "Jira/Zephyr — Atlassian import style",
    "TestRail — TestRail format",
    "Cucumber/BDD — Gherkin syntax",
    "Excel/CSV — Generic import format",
    "Custom — I will define",
]

def fill_sample_data():
    st.session_state.px_platform = "Web"
    st.session_state.px_context = "zepto.com — user added a new address"
    st.session_state.px_requirements = "User logs in with valid credentials, adds a new address, and sees it saved correctly."
    st.session_state.px_additional = "Consider invalid address fields, network timeouts, duplicates, and location pin issues."
    st.session_state.px_role = TESTING_ROLES[0]
    st.session_state.px_test_type = "Functional Testing"
    st.session_state.px_format = TEST_MGMT_FORMATS[0]
    st.session_state.px_custom_format_notes = ""
    st.session_state.px_cases = 20
    st.session_state.px_id_prefix = "TC"
    st.session_state.px_priority = "P0/P1/P2"
    st.session_state.px_include_comprehensive = True
    st.session_state.px_include_edge = True
    st.session_state.px_include_negative = True
    st.session_state.px_include_boundary = True
    st.session_state.px_include_ui = True
    st.session_state.px_include_api = False
    st.session_state.px_include_accessibility = False
    st.session_state.px_include_security = False
    st.session_state.px_include_performance = False

def build_advanced_prompt(cfg: dict) -> str:
    role = cfg["role"]
    test_type = cfg["test_type"]
    fmt = cfg["format"]
    platform = cfg.get("platform", "Web")
    context = cfg["context"].strip()
    req = cfg["requirements"].strip()
    add = cfg["additional"].strip()

    n_cases = cfg["n_cases"]
    id_prefix = cfg["id_prefix"].strip() or "TC"
    priority_scheme = cfg["priority_scheme"]
    custom_notes = (cfg.get("custom_format_notes") or "").strip()

    flags = []
    if cfg["include_comprehensive"]: flags.append("Comprehensive coverage")
    if cfg["include_edge"]: flags.append("Edge cases")
    if cfg["include_negative"]: flags.append("Negative tests")
    if cfg["include_boundary"]: flags.append("Boundary/validation tests")
    if cfg["include_ui"]: flags.append("UI validations")
    if cfg["include_api"]: flags.append("API validations")
    if cfg["include_accessibility"]: flags.append("Accessibility basics (WCAG)")
    if cfg["include_security"]: flags.append("Security basics (OWASP)")
    if cfg["include_performance"]: flags.append("Performance basics (latency/throughput)")
    focus = ", ".join(flags) if flags else "Functional coverage"

    # Output instructions by format
    if fmt.startswith("Cucumber/BDD"):
        output_instructions = (
            "Output strictly in BDD Gherkin:\n"
            "- Provide a Feature title\n"
            "- Write 8–15 Scenarios with Given/When/Then\n"
            "- Include tags like @smoke @regression @negative where relevant\n"
        )
    elif fmt.startswith("Excel/CSV"):
        output_instructions = (
            "Output as CSV in plain text (comma-separated) with header:\n"
            "TestCaseID,Title,Preconditions,Steps,ExpectedResult,Priority,Type\n"
        )
    elif fmt.startswith("Jira/Zephyr"):
        output_instructions = (
            "Output as a Markdown table with columns:\n"
            "Test Case ID | Title | Preconditions | Steps | Expected Result | Priority | Type\n"
            "Make titles concise and import-friendly for Jira/Zephyr. Keep steps structured.\n"
        )
    elif fmt.startswith("TestRail"):
        output_instructions = (
            "Output as a Markdown table with columns:\n"
            "Test Case ID | Title | Preconditions | Steps | Expected Result | Priority | Type\n"
            "Make steps explicit and structured for TestRail.\n"
        )
    elif fmt.startswith("Custom"):
        output_instructions = (
            "Output in the custom format described below. If anything is unclear, pick the closest reasonable structure.\n"
            f"CUSTOM FORMAT NOTES:\n{custom_notes if custom_notes else '(No custom notes provided — use a clean detailed table format.)'}\n"
        )
    else:
        output_instructions = (
            "Output as a Markdown table with columns:\n"
            "Test Case ID | Title | Preconditions | Steps | Expected Result | Priority | Type\n"
        )

    prompt = f"""
You are acting as: {role}

TASK:
Create {n_cases} high-quality QA test cases for: {test_type}

PLATFORM / SURFACE:
{platform}

CONTEXT (What are we testing?):
{context}

REQUIREMENTS (What should it do?):
{req}

ADDITIONAL INFORMATION:
{add if add else "(none)"}

COVERAGE FOCUS:
{focus}

QUALITY BAR:
- Mix functional + negative + edge/boundary where applicable
- Include validations, error messaging, state handling, and idempotency where relevant
- Avoid hallucinating features that are not implied by the requirements
- Clearly state assumptions only when necessary

PRIORITY:
Use {priority_scheme}

TEST CASE ID:
Use IDs like {id_prefix}-001, {id_prefix}-002, ... sequentially.

OUTPUT FORMAT (STRICT):
{output_instructions}

IMPORTANT:
- Do not include any real secrets or private data.
- Keep results demo-friendly and export-ready.
""".strip()

    return prompt

# =========================================================
# CLIPBOARD (Copy buttons) — JS component
# =========================================================
def clipboard_button(label: str, text_to_copy: str, button_id: str):
    # Use a small HTML/JS snippet for copy-to-clipboard
    safe_text = (text_to_copy or "").replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <div style="margin: 6px 0 10px 0;">
      <button id="{button_id}"
        style="
          padding:10px 14px;
          border-radius:12px;
          border:1px solid rgba(255,255,255,0.14);
          background: rgba(255,255,255,0.06);
          color: rgba(255,255,255,0.90);
          cursor:pointer;
          width: 100%;
        "
      >{label}</button>
      <script>
        const btn = document.getElementById("{button_id}");
        if(btn) {{
          btn.onclick = async () => {{
            try {{
              await navigator.clipboard.writeText(`{safe_text}`);
              btn.innerText = "✅ Copied!";
              setTimeout(()=>btn.innerText="{label}", 1400);
            }} catch (e) {{
              btn.innerText = "❌ Copy failed";
              setTimeout(()=>btn.innerText="{label}", 1400);
            }}
          }};
        }}
      </script>
    </div>
    """
    st.components.v1.html(html, height=60)

# =========================================================
# LOGIN UI (KEEP)
# =========================================================
def render_footer():
    st.markdown(
        """
<div class="px-footer">
  Thought and built by
  <a href="https://www.linkedin.com/in/monika-kushwaha-52443735/" target="_blank"><b>Monika Kushwaha</b></a><br/>
  QA Engineer | GenAI Product Management | LLMs, RAG, Automation, Performance
</div>
""",
        unsafe_allow_html=True,
    )

def render_login():
    st.markdown('<div class="center-auth"><div class="px-card">', unsafe_allow_html=True)
    st.markdown('<div class="px-h2">🔐 Login to Promptix</div>', unsafe_allow_html=True)
    st.markdown('<div class="px-muted">Login is required to enforce accurate daily free limits.</div>', unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Login", "Create account"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Please enter email and password.")
            else:
                data, err = supabase_sign_in(email.strip(), password)
                if err:
                    st.error(f"Login failed: {err}")
                else:
                    user_obj = data.get("user") or {}
                    access_token = data.get("access_token", "")
                    refresh_token = data.get("refresh_token", "")
                    if not user_obj.get("id") or not access_token or not refresh_token:
                        st.error("Login succeeded but tokens are missing. Check Supabase Auth settings.")
                    else:
                        apply_auth_session(user_obj, access_token, refresh_token)
                        st.success("Logged in ✅")
                        st.rerun()

    with tab_signup:
        with st.form("signup_form"):
            email2 = st.text_input("Email (new account)", placeholder="you@example.com")
            password2 = st.text_input("Password", type="password")
            submit2 = st.form_submit_button("Create account", use_container_width=True)

        if submit2:
            if not email2 or not password2:
                st.error("Please enter email and password.")
            else:
                data, err = supabase_sign_up(email2.strip(), password2)
                if err:
                    st.error(f"Signup failed: {err}")
                else:
                    access_token = (data or {}).get("access_token", "")
                    refresh_token = (data or {}).get("refresh_token", "")
                    user_obj = (data or {}).get("user") or {}
                    if access_token and refresh_token and user_obj.get("id"):
                        apply_auth_session(user_obj, access_token, refresh_token)
                        st.success("Account created & logged in ✅")
                        st.rerun()
                    else:
                        st.success("Account created ✅")
                        st.info("If email confirmation is enabled in Supabase, confirm your email then login.")

    st.markdown("</div></div>", unsafe_allow_html=True)
    render_footer()

# =========================================================
# PROVIDERS (RESTORED)
# =========================================================
PROVIDERS = [
    "Promptix Free (LLaMA via Together)",
    "Together LLaMA (BYOK)",
    "OpenAI (BYOK)",
    "Google Gemini (BYOK)",
    "Anthropic Claude (BYOK)",
]

def render_sidebar(user, used_free: int, remaining_free: int):
    st.sidebar.markdown("## 🔐 AI Settings")
    provider = st.sidebar.selectbox("Provider", PROVIDERS, index=0, key="px_provider")

    cfg = {"provider": provider}

    # Common: show endpoint where relevant (as per your older UI)
    if provider in ("Promptix Free (LLaMA via Together)", "Together LLaMA (BYOK)"):
        endpoint = st.sidebar.text_input("API Endpoint", value=TOGETHER_BASE_URL_DEFAULT, key="px_together_endpoint")
        cfg["base_url"] = endpoint.strip()

        # model: provide select + custom
        model_choice = st.sidebar.selectbox(
            "Model (Together)",
            options=["(Recommended) " + TOGETHER_MODEL_SAFE_DEFAULT, "(Try) " + TOGETHER_MODEL_ALT_70B, "Custom…"],
            index=0,
            key="px_together_model_choice",
        )
        if model_choice == "Custom…":
            model = st.sidebar.text_input("Custom model id", value=TOGETHER_MODEL_SAFE_DEFAULT, key="px_together_model_custom")
        else:
            model = model_choice.replace("(Recommended) ", "").replace("(Try) ", "").strip()
        cfg["model"] = model

    if provider == "Promptix Free (LLaMA via Together)":
        cfg["api_key"] = TOGETHER_API_KEY_SERVER

        st.sidebar.markdown(
            f"""
<div class="kpi-box">
<b>Free calls left today:</b><br/>
{remaining_free}/{DAILY_FREE_LIMIT}
</div>
""",
            unsafe_allow_html=True,
        )
        if TOGETHER_API_KEY_SERVER:
            st.sidebar.markdown(
                """
<div class="kpi-green">
<b>Using server key from env var:</b><br/>
TOGETHER_API_KEY
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.warning("TOGETHER_API_KEY is missing in Secrets. Free mode won't work.")

    elif provider == "Together LLaMA (BYOK)":
        cfg["api_key"] = st.sidebar.text_input("Together API Key", type="password", placeholder="Paste your key", key="px_together_key")

    elif provider == "OpenAI (BYOK)":
        cfg["base_url"] = st.sidebar.text_input("API Endpoint", value=OPENAI_BASE_URL_DEFAULT, key="px_openai_base").strip()
        cfg["model"] = st.sidebar.text_input("Model", value=OPENAI_MODEL_DEFAULT, key="px_openai_model")
        cfg["api_key"] = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="Paste your key", key="px_openai_key")

    elif provider == "Google Gemini (BYOK)":
        cfg["model"] = st.sidebar.text_input("Model", value=GEMINI_MODEL_DEFAULT, key="px_gemini_model")
        cfg["api_key"] = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Paste your key", key="px_gemini_key")

    elif provider == "Anthropic Claude (BYOK)":
        cfg["model"] = st.sidebar.text_input("Model", value=ANTHROPIC_MODEL_DEFAULT, key="px_anthropic_model")
        cfg["api_key"] = st.sidebar.text_input("Anthropic API Key", type="password", placeholder="Paste your key", key="px_anthropic_key")

    st.session_state.llm_cfg = cfg

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Get your API keys")
    st.sidebar.markdown(
        """
<div class="px-links">
• <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI: Get your API key</a><br/>
• <a href="https://aistudio.google.com/app/apikey" target="_blank">Gemini: Get your API key</a><br/>
• <a href="https://console.anthropic.com/" target="_blank">Anthropic: Get your API key</a><br/>
• <a href="https://api.together.xyz/settings/api-keys" target="_blank">Together: Get your API key</a><br/>
</div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Logged in:**")
    st.sidebar.markdown(user.get("email", ""))
    if st.sidebar.button("Logout", use_container_width=True):
        do_logout()

def send_to_ai(prompt: str) -> str:
    cfg = st.session_state.get("llm_cfg") or {}
    provider = cfg.get("provider")

    if provider in ("Promptix Free (LLaMA via Together)", "Together LLaMA (BYOK)", "OpenAI (BYOK)"):
        if not cfg.get("api_key"):
            raise RuntimeError("Missing API key.")
        return call_openai_compatible(cfg.get("base_url") or OPENAI_BASE_URL_DEFAULT, cfg["api_key"], cfg.get("model") or OPENAI_MODEL_DEFAULT, prompt)

    if provider == "Google Gemini (BYOK)":
        if not cfg.get("api_key"):
            raise RuntimeError("Missing Gemini API key.")
        return call_gemini(cfg["api_key"], cfg.get("model") or GEMINI_MODEL_DEFAULT, prompt)

    if provider == "Anthropic Claude (BYOK)":
        if not cfg.get("api_key"):
            raise RuntimeError("Missing Anthropic API key.")
        return call_anthropic(cfg["api_key"], cfg.get("model") or ANTHROPIC_MODEL_DEFAULT, prompt)

    raise RuntimeError("Invalid provider.")

# =========================================================
# MAIN APP
# =========================================================
def render_main_app():
    user = st.session_state.user
    access_token = st.session_state.access_token
    usage_date = today_local_iso()

    used_free = get_daily_usage(user["id"], access_token, usage_date)
    remaining_free = max(0, DAILY_FREE_LIMIT - used_free)

    render_sidebar(user, used_free, remaining_free)

    # HERO
    st.markdown(
        """
<div class="hero">
  <div style="font-size:30px;font-weight:800;">
    Promptix AI v2 <span class="badge">MVP</span>
  </div>
  <div class="subtle">
    Turn product requirements into structured, export-ready test cases — with edge cases, negatives, and multiple formats (Jira/Zephyr/TestRail/BDD/CSV).
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    # Header strip (like your earlier KPI row)
    col1, col2, col3 = st.columns([2.4, 1, 1])
    with col1:
        st.info(f"👤 Logged in as **{user.get('email','')}**")
    with col2:
        st.metric("Daily free limit", f"{DAILY_FREE_LIMIT}/day")
    with col3:
        st.metric("Remaining today", str(remaining_free))

    st.markdown("## 🧪 Test Configuration")

    # Sample data button
    if st.button("🎯 Fill Sample Data", use_container_width=True):
        fill_sample_data()

    # Defaults
    st.session_state.setdefault("px_platform", "Web")
    st.session_state.setdefault("px_role", TESTING_ROLES[0])
    st.session_state.setdefault("px_test_type", TEST_TYPES[0])
    st.session_state.setdefault("px_format", TEST_MGMT_FORMATS[0])
    st.session_state.setdefault("px_custom_format_notes", "")

    st.session_state.setdefault("px_context", "")
    st.session_state.setdefault("px_requirements", "")
    st.session_state.setdefault("px_additional", "")

    st.session_state.setdefault("px_cases", 20)
    st.session_state.setdefault("px_id_prefix", "TC")
    st.session_state.setdefault("px_priority", "P0/P1/P2")

    st.session_state.setdefault("px_include_comprehensive", True)
    st.session_state.setdefault("px_include_edge", True)
    st.session_state.setdefault("px_include_negative", True)
    st.session_state.setdefault("px_include_boundary", True)
    st.session_state.setdefault("px_include_ui", True)
    st.session_state.setdefault("px_include_api", False)
    st.session_state.setdefault("px_include_accessibility", False)
    st.session_state.setdefault("px_include_security", False)
    st.session_state.setdefault("px_include_performance", False)

    # Core fields
    st.selectbox("Testing Role", TESTING_ROLES, key="px_role")
    st.selectbox("Test Type", TEST_TYPES, key="px_test_type")
    st.selectbox("Test Management Format (Import-Ready)", TEST_MGMT_FORMATS, key="px_format")

    if str(st.session_state.px_format).startswith("Custom"):
        st.text_area("Custom format notes (optional)", key="px_custom_format_notes", height=90, max_chars=2000)

    st.selectbox("Platform", PLATFORMS, key="px_platform")

    st.text_area("Context (What are you testing?)", key="px_context", height=120, max_chars=4000)
    st.text_area("Requirements (What should it do?)", key="px_requirements", height=140, max_chars=4000)
    st.text_area("Additional Information", key="px_additional", height=120, max_chars=4000)

    st.markdown(
        """
<div class="callout">
<b>Important</b>
<ul style="margin:8px 0 0 18px;">
  <li>Avoid real secrets or client data</li>
  <li>Review AI output before use</li>
  <li>Use as assistant — not the only source of truth</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    # Checkboxes
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("✅ Comprehensive Coverage", key="px_include_comprehensive")
        st.checkbox("🧪 Include Edge Cases", key="px_include_edge")
        st.checkbox("❌ Include Negative Tests", key="px_include_negative")
    with c2:
        st.checkbox("📏 Include Boundary/Validation Tests", key="px_include_boundary")
        st.checkbox("🖥️ Include UI Validations", key="px_include_ui")
        st.checkbox("🔌 Include API Validations", key="px_include_api")
    with c3:
        st.checkbox("♿ Accessibility (basic)", key="px_include_accessibility")
        st.checkbox("🛡️ Security (basic)", key="px_include_security")
        st.checkbox("⚡ Performance (basic)", key="px_include_performance")

    # Advanced Prompt Engineering
    st.write("")
    with st.expander("⚡ Advanced Prompt Engineering", expanded=True):
        st.slider("Number of test cases", min_value=8, max_value=40, value=int(st.session_state.px_cases), key="px_cases")
        st.text_input("Test Case ID prefix", value=st.session_state.px_id_prefix, key="px_id_prefix")
        st.selectbox("Priority scheme", ["P0/P1/P2", "High/Medium/Low", "Critical/Major/Minor"], index=0, key="px_priority")

    st.write("")
    colA, colB = st.columns([1, 1])

    # Generate prompt
    with colA:
        if st.button("⚡ Generate Advanced Prompt", use_container_width=True):
            if not st.session_state.px_context.strip() or not st.session_state.px_requirements.strip():
                st.error("Please fill Context and Requirements first.")
            else:
                prompt_cfg = {
                    "role": st.session_state.px_role,
                    "test_type": st.session_state.px_test_type,
                    "format": st.session_state.px_format,
                    "custom_format_notes": st.session_state.get("px_custom_format_notes", ""),
                    "platform": st.session_state.px_platform,
                    "context": st.session_state.px_context,
                    "requirements": st.session_state.px_requirements,
                    "additional": st.session_state.px_additional,
                    "n_cases": int(st.session_state.px_cases),
                    "id_prefix": st.session_state.px_id_prefix,
                    "priority_scheme": st.session_state.px_priority,
                    "include_comprehensive": st.session_state.px_include_comprehensive,
                    "include_edge": st.session_state.px_include_edge,
                    "include_negative": st.session_state.px_include_negative,
                    "include_boundary": st.session_state.px_include_boundary,
                    "include_ui": st.session_state.px_include_ui,
                    "include_api": st.session_state.px_include_api,
                    "include_accessibility": st.session_state.px_include_accessibility,
                    "include_security": st.session_state.px_include_security,
                    "include_performance": st.session_state.px_include_performance,
                }
                st.session_state.advanced_prompt = build_advanced_prompt(prompt_cfg)
                st.success("Advanced prompt generated ✅")

    # Send to AI
    with colB:
        can_send = bool(st.session_state.get("advanced_prompt"))
        if st.button("🚀 Send to AI", use_container_width=True, disabled=not can_send):
            cfg = st.session_state.get("llm_cfg") or {}
            provider = cfg.get("provider")

            # Enforce free limit only for Promptix Free
            if provider == "Promptix Free (LLaMA via Together)":
                if remaining_free <= 0:
                    st.error("Daily free limit reached. Please come back tomorrow or use BYOK.")
                    return
                if not TOGETHER_API_KEY_SERVER:
                    st.error("TOGETHER_API_KEY is missing in Secrets. Free mode can't run.")
                    return

            # BYOK key required
            if provider != "Promptix Free (LLaMA via Together)" and not cfg.get("api_key"):
                st.error("Please add your API key in the sidebar.")
                return

            # Best-effort model check for Together to reduce the error you saw
            if provider in ("Promptix Free (LLaMA via Together)", "Together LLaMA (BYOK)"):
                key_for_check = cfg.get("api_key") or ""
                model_set = together_list_models(key_for_check) if key_for_check else None
                chosen_model = cfg.get("model") or ""
                if model_set is not None and chosen_model and chosen_model not in model_set:
                    st.warning(
                        f"Selected model may not be available for this key. "
                        f"Try '{TOGETHER_MODEL_SAFE_DEFAULT}' or pick from Together models."
                    )

            with st.spinner("Sending to AI…"):
                try:
                    output = send_to_ai(st.session_state.advanced_prompt)
                except Exception as e:
                    msg = str(e)

                    # Auto-fix the common Together model error you showed
                    if "Unable to access model" in msg and "together" in (cfg.get("base_url") or "").lower():
                        st.error(msg)
                        st.info(
                            "Fix: your Together key can't access this model. "
                            f"Switching to safer default: {TOGETHER_MODEL_SAFE_DEFAULT}"
                        )
                        # force safer model for Together
                        if provider in ("Promptix Free (LLaMA via Together)", "Together LLaMA (BYOK)"):
                            st.session_state["px_together_model_choice"] = "(Recommended) " + TOGETHER_MODEL_SAFE_DEFAULT
                            st.session_state.llm_cfg["model"] = TOGETHER_MODEL_SAFE_DEFAULT
                        return

                    st.error(f"AI request failed: {msg}")
                    return

            # Count usage after success (free only)
            if provider == "Promptix Free (LLaMA via Together)":
                new_used = increment_daily_usage(user["id"], access_token, usage_date)
                new_remaining = max(0, DAILY_FREE_LIMIT - new_used)
                st.success(f"Done ✅  Free calls left today: {new_remaining}/{DAILY_FREE_LIMIT}")
            else:
                st.success("Done ✅")

            st.session_state.last_output = output

    # PROMPT (Bigger box) + Copy
    st.write("")
    st.subheader("🧾 Advanced Prompt (Editable)")

    st.text_area(
        label="You can edit the prompt before sending",
        key="advanced_prompt",
        height=420,  # bigger (your complaint)
        placeholder="Click 'Generate Advanced Prompt' to create one…",
    )

    if st.session_state.get("advanced_prompt"):
        clipboard_button("📋 Copy Advanced Prompt", st.session_state.get("advanced_prompt", ""), "copy_prompt_btn")

    # OUTPUT + Copy answer (restored) + Download
    if st.session_state.get("last_output"):
        st.write("")
        st.subheader("✅ AI Response")

        # show output (markdown) + also provide raw text for copy
        st.markdown(st.session_state.last_output)

        cdl1, cdl2 = st.columns([1, 1])
        with cdl1:
            clipboard_button("📋 Copy AI Response", st.session_state.get("last_output", ""), "copy_output_btn")
        with cdl2:
            st.download_button(
                "⬇️ Download Output (.md)",
                data=st.session_state.last_output.encode("utf-8"),
                file_name=f"promptix_output_{usage_date}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    render_footer()

# =========================================================
# ENTRY
# =========================================================
try_bootstrap_session_from_cookie()

if not is_authed():
    render_login()
    st.stop()

render_main_app()
