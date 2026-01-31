# Promptix AI v2 — Streamlit single-file app
# ✅ SAME UI + SAME functionality
# ✅ Fixes ONLY:
#   - cookie duplicate key issue (StreamlitDuplicateElementKey) + login/logout flicker
#   - decrement daily free limit ONLY on "Send to AI"
#   - popup after 3 calls + block further calls once limit reached

import os
import json
import time
import uuid
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import requests
import streamlit as st

try:
    import extra_streamlit_components as stx
except Exception:
    stx = None


# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="Promptix AI v2", layout="wide")

DAILY_FREE_LIMIT = 3
COOKIE_NAME = "px_session_v2"

DEFAULT_TOGETHER_BASE = "https://api.together.xyz/v1/chat/completions"

# Together free mode server key (set in Streamlit secrets / env)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# Supabase (login + usage persistence)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

USAGE_TABLE = os.getenv("PROMPTIX_USAGE_TABLE", "promptix_usage")  # (user_id, usage_date, count, updated_at)


# ==========================================================
# DATA MODELS
# ==========================================================
@dataclass
class Provider:
    key: str
    label: str
    vendor: str  # together/openai/gemini/anthropic
    kind: str    # free/byok
    needs_key: bool


PROVIDERS: List[Provider] = [
    Provider(key="promptix_free", label="Promptix Free (LLAMA via Together)", vendor="together", kind="free", needs_key=False),
    Provider(key="openai", label="OpenAI (BYOK)", vendor="openai", kind="byok", needs_key=True),
    Provider(key="gemini", label="Gemini (BYOK)", vendor="gemini", kind="byok", needs_key=True),
    Provider(key="anthropic", label="Anthropic (BYOK)", vendor="anthropic", kind="byok", needs_key=True),
    Provider(key="together_byok", label="Together (BYOK)", vendor="together", kind="byok", needs_key=True),
]

# Together models (kept)
TOGETHER_MODELS: List[Tuple[str, str]] = [
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
]

OPENAI_MODELS: List[Tuple[str, str]] = [
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4o", "gpt-4o"),
]

GEMINI_MODELS: List[Tuple[str, str]] = [
    ("gemini-1.5-flash", "gemini-1.5-flash"),
    ("gemini-1.5-pro", "gemini-1.5-pro"),
]

ANTHROPIC_MODELS: List[Tuple[str, str]] = [
    ("claude-3-5-sonnet-latest", "claude-3-5-sonnet-latest"),
    ("claude-3-5-haiku-latest", "claude-3-5-haiku-latest"),
]


# ==========================================================
# HELPERS
# ==========================================================
def toast(msg: str, icon: str = "✅"):
    try:
        st.toast(msg, icon=icon)
    except Exception:
        # older streamlit fallback
        st.info(msg)


def now_local_date_str() -> str:
    # daily reset by local date (server timezone). Good enough for demo.
    return dt.datetime.now().strftime("%Y-%m-%d")


def stable_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def clipboard_button(label: str, text: str, btn_key: str):
    # Small JS clipboard helper, keeps UI same
    if st.button(label, key=btn_key):
        st.session_state["_copy_payload"] = text
        toast("Copied ✅", icon="📋")

    # Execute copy once
    if st.session_state.get("_copy_payload") and st.session_state.get("_copy_payload") == text:
        st.markdown(
            f"""
<script>
navigator.clipboard.writeText({json.dumps(text)});
</script>
""",
            unsafe_allow_html=True,
        )
        st.session_state["_copy_payload"] = None


# ==========================================================
# COOKIE SESSION (ONE cookie call per run)
# ==========================================================
cookie_manager = stx.CookieManager() if stx else None


def queue_cookie_set(value: str):
    st.session_state["_cookie_action"] = {"op": "set", "value": value}


def queue_cookie_delete():
    st.session_state["_cookie_action"] = {"op": "delete"}


def process_cookie_queue_once():
    """
    Execute at most ONE cookie write/delete per run.
    Prevents StreamlitDuplicateElementKey + flicker during login/logout.
    """
    if not cookie_manager:
        return
    action = st.session_state.get("_cookie_action")
    if not action:
        return
    try:
        if action["op"] == "set":
            value = action["value"]
            cookie_manager.set(COOKIE_NAME, value)
        elif action["op"] == "delete":
            cookie_manager.delete(COOKIE_NAME)
    finally:
        st.session_state.pop("_cookie_action", None)
        st.rerun()


def bootstrap_session_from_cookie_once():
    """
    Read cookie ONCE per run (and never in the same run as a cookie write/delete).
    """
    if not cookie_manager:
        return
    if st.session_state.get("_cookie_bootstrapped"):
        return
    if st.session_state.get("_cookie_action"):
        # If a cookie op is queued, skip reads this run.
        st.session_state["_cookie_bootstrapped"] = True
        return

    st.session_state["_cookie_bootstrapped"] = True
    try:
        raw = cookie_manager.get(COOKIE_NAME)
        if not raw:
            return
        data = json.loads(raw)
        if "user" in data and "access_token" in data and "refresh_token" in data:
            st.session_state["user"] = data["user"]
            st.session_state["access_token"] = data["access_token"]
            st.session_state["refresh_token"] = data["refresh_token"]
    except Exception:
        # ignore corrupt cookies
        return


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
def usage_table_headers(access_token: str) -> Dict[str, str]:
    # Use the logged-in user's JWT as bearer for RLS
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def get_daily_used_count(user_id: str, usage_date: str) -> int:
    """
    Returns MAX(count) for (user_id, usage_date). If table has multiple rows, MAX handles it safely.
    """
    try:
        access_token = st.session_state.get("access_token", "")
        if not (SUPABASE_URL and SUPABASE_ANON_KEY and access_token):
            return 0

        url = f"{SUPABASE_URL}/rest/v1/{USAGE_TABLE}?user_id=eq.{user_id}&usage_date=eq.{usage_date}&select=count"
        r = requests.get(url, headers=usage_table_headers(access_token), timeout=20)
        if r.status_code >= 300:
            return 0
        rows = r.json() or []
        if not rows:
            return 0
        return int(max(int(x.get("count", 0) or 0) for x in rows))
    except Exception:
        return 0


def increment_daily_usage(user_id: str, usage_date: str) -> int:
    """
    Atomic-ish for demo: read current MAX(count), then upsert a new row with count+1.
    """
    current = get_daily_used_count(user_id, usage_date)
    new_count = current + 1
    try:
        access_token = st.session_state.get("access_token", "")
        if not (SUPABASE_URL and SUPABASE_ANON_KEY and access_token):
            return current

        payload = {
            "user_id": user_id,
            "usage_date": usage_date,
            "count": new_count,
            "updated_at": dt.datetime.utcnow().isoformat(),
        }
        url = f"{SUPABASE_URL}/rest/v1/{USAGE_TABLE}"
        # Upsert via PostgREST with on_conflict if you have a unique constraint on (user_id, usage_date).
        # Even without constraint, MAX(count) still works.
        r = requests.post(url, headers=usage_table_headers(access_token), json=payload, timeout=20)
        if r.status_code >= 300:
            return current
        return new_count
    except Exception:
        return current


# ==========================================================
# LLM CALLS
# ==========================================================
def call_together_chat(api_key: str, model: str, messages: List[Dict[str, str]], base_url: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1800,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(base_url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(r.text[:1200])
    data = r.json()
    return data["choices"][0]["message"]["content"]


def call_openai_chat(api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": 0.4}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(r.text[:1200])
    data = r.json()
    return data["choices"][0]["message"]["content"]


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    # Gemini REST
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.4}}
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(r.text[:1200])
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_anthropic(api_key: str, model: str, prompt: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1800,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 300:
        raise RuntimeError(r.text[:1200])
    data = r.json()
    # Anthropics returns content array
    return "".join(part.get("text", "") for part in data.get("content", []))


# ==========================================================
# PROMPT BUILDER (kept same checkboxes + advanced prompt)
# ==========================================================
TESTING_ROLES = [
    "Automation Tester – Test automation specialist",
    "Manual QA – Functional tester",
    "QA Lead – Test strategy & planning",
    "SDET – Framework + CI/CD quality",
    "Performance QA – Load & reliability",
    "Security QA – AppSec validation",
]

TEST_TYPES = [
    "Regression Testing",
    "Integration Testing",
    "Smoke Testing",
    "Sanity Testing",
    "UAT Testing",
    "Exploratory Testing",
]

OUTPUT_FORMATS = [
    "Cucumber/BDD – Gherkin syntax",
    "Jira – Test case format",
    "TestRail – Steps/Expected format",
    "CSV – Spreadsheet style",
    "Plain – Structured bullets",
]

# Coverage flags (checkbox list)
COVERAGE_FLAGS = [
    "Comprehensive coverage",
    "Include edge cases",
    "Include negative test cases",
    "API validation tests",
    "Security checks",
    "Performance checks",
    "Accessibility checks",
    "Cross-browser/device coverage",
    "Data validation",
    "Error handling & fallback",
]


def build_advanced_prompt(data: Dict[str, Any]) -> str:
    flags = data.get("coverage_flags", [])
    flags_str = ", ".join(flags) if flags else "Standard coverage"

    return f"""You are Promptix AI v2 — a senior QA engineer + test architect.

TASK
Generate high-quality test cases from the given product context and requirements.

TESTER ROLE: {data.get("role")}
TEST TYPE: {data.get("test_type")}
OUTPUT FORMAT: {data.get("output_format")}
COVERAGE FLAGS: {flags_str}

WHAT YOU ARE TESTING (CONTEXT)
{data.get("context")}

WHAT IT SHOULD DO (REQUIREMENTS / ACCEPTANCE CRITERIA)
{data.get("requirements")}

ADDITIONAL INFORMATION / CONSTRAINTS
{data.get("additional_info")}

EDGE CASES (IF ANY)
{data.get("edge_cases")}

NON-FUNCTIONAL REQUIREMENTS (IF ANY)
{data.get("nfr")}

OUTPUT RULES
- Make it import-ready and structured
- Include steps, expected results, and test data where relevant
- Avoid fluff; be precise
"""


# ==========================================================
# STYLES (kept)
# ==========================================================
def inject_css():
    st.markdown(
        """
<style>
:root {
  --px-bg: #0b0f16;
  --px-card: rgba(255,255,255,0.06);
  --px-border: rgba(255,255,255,0.10);
  --px-text: rgba(255,255,255,0.92);
  --px-muted: rgba(255,255,255,0.65);
  --px-purple: #6d46ff;
  --px-accent: #ff3b3b;
}

html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 700px at 30% 15%, rgba(109,70,255,0.35), transparent 60%),
              radial-gradient(1000px 700px at 75% 30%, rgba(0,180,255,0.22), transparent 55%),
              var(--px-bg) !important;
  color: var(--px-text) !important;
}

.block-container { padding-top: 1.2rem; }

.px-hero {
  padding: 24px 22px;
  border-radius: 18px;
  background: linear-gradient(90deg, rgba(109,70,255,0.55), rgba(109,70,255,0.15));
  border: 1px solid rgba(255,255,255,0.10);
  margin-bottom: 18px;
}

.px-hero h1 { margin:0; font-size: 32px; line-height: 1.2; }
.px-hero p { margin:10px 0 0 0; color: var(--px-muted); font-size: 15px; }

.px-card {
  background: var(--px-card);
  border: 1px solid var(--px-border);
  border-radius: 16px;
  padding: 16px 16px;
}

.px-muted { color: var(--px-muted); font-size: 13px; }

.px-row {
  display:flex; gap:14px; flex-wrap:wrap;
}
.px-stat {
  flex: 1;
  min-width: 180px;
  background: var(--px-card);
  border: 1px solid var(--px-border);
  border-radius: 16px;
  padding: 14px 14px;
}
.px-stat b { display:block; margin-top:6px; font-size: 22px; }

.px-footer {
  margin-top: 28px;
  padding: 16px 10px;
  text-align:center;
  color: var(--px-muted);
  font-size: 13px;
}

a { color: #56c2ff !important; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Buttons */
.stButton>button {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
.stButton>button:hover {
  border-color: rgba(255,255,255,0.22) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(0,0,0,0.25) !important;
  border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# UI — AUTH
# ==========================================================
def render_login():
    st.markdown(
        """
<div class="center-auth">
  <div class="px-card" style="max-width:980px; margin:0 auto;">
    <h2 style="margin-top:0;">🔐 Login to Promptix</h2>
    <div class="px-muted">Login is required to enforce accurate daily free limits.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

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

    # Placeholders (so we can update limits live after Send to AI without changing UI)
    limit_card_slot = None
    hero_remaining_slot = None

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
                    st.warning("TOGETHER_API_KEY is not configured on the server.")
                api_key = TOGETHER_API_KEY

                limit_card_slot = st.empty()
                limit_card_slot.markdown(f"""
<div class="px-card" style="margin-top:10px;">
  <div style="font-weight:700;">Free calls left today:</div>
  <div style="font-size:1.25rem; margin-top:6px;">{remaining}/{DAILY_FREE_LIMIT}</div>
  <div class="px-muted">Used today: {used}</div>
</div>
""", unsafe_allow_html=True)
                st.markdown(
                    """
<div class="px-card" style="margin-top:10px; background: rgba(18,120,60,0.25); border-color: rgba(18,120,60,0.35);">
  <div style="font-weight:700;">Using server key from env:</div>
  <div class="px-muted" style="margin-top:6px;">TOGETHER_API_KEY</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            else:
                api_key = st.text_input("Together API Key (BYOK)", type="password", key="together_key")
                st.caption("Security: never hardcode or commit keys to GitHub.")

        elif provider.vendor == "openai":
            model_keys = [m[0] for m in OPENAI_MODELS]
            model_labels = [m[1] for m in OPENAI_MODELS]
            selected_m = st.selectbox("Model (OpenAI)", model_labels, index=0, key="openai_model_label")
            model = model_keys[model_labels.index(selected_m)]
            api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")

        elif provider.vendor == "gemini":
            model_keys = [m[0] for m in GEMINI_MODELS]
            model_labels = [m[1] for m in GEMINI_MODELS]
            selected_m = st.selectbox("Model (Gemini)", model_labels, index=0, key="gemini_model_label")
            model = model_keys[model_labels.index(selected_m)]
            api_key = st.text_input("Gemini API Key", type="password", key="gemini_key")

        elif provider.vendor == "anthropic":
            model_keys = [m[0] for m in ANTHROPIC_MODELS]
            model_labels = [m[1] for m in ANTHROPIC_MODELS]
            selected_m = st.selectbox("Model (Anthropic)", model_labels, index=0, key="anthropic_model_label")
            model = model_keys[model_labels.index(selected_m)]
            api_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")

        st.markdown("---")
        st.markdown("🔗 **Get your API key**")
        st.markdown("- OpenAI: [Get your API key](https://platform.openai.com/api-keys)")
        st.markdown("- Gemini: [Get your API key](https://aistudio.google.com/app/apikey)")
        st.markdown("- Anthropic: [Get your API key](https://console.anthropic.com/settings/keys)")
        st.markdown("- Together: [Get your API key](https://api.together.xyz/settings/api-keys)")

        st.markdown("---")
        st.caption(f"Logged in as: **{email}**")
        if st.button("Logout", use_container_width=True):
            clear_auth_session()
            toast("Logged out ✅", icon="👋")
            st.rerun()

    # Hero
    st.markdown(
        """
<div class="px-hero">
  <h1>Promptix AI v2 <span style="font-size:13px; background: rgba(255,255,255,0.12); padding: 4px 10px; border-radius: 999px; margin-left:10px;">MVP</span></h1>
  <p>Turn product requirements into structured, export-ready test cases — with edge cases, negatives, and multiple formats (Jira/TestRail/BDD/CSV).</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Mini stats row (kept)
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        st.markdown("<div class='px-stat'><div class='px-muted'>Daily free limit</div><b>3/day</b></div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='px-stat'><div class='px-muted'>Provider</div><b>Free + BYOK</b></div>", unsafe_allow_html=True)
    with colC:
        hero_remaining_slot = st.empty()
        hero_remaining_slot.markdown(
            f"<div class='px-muted'>Remaining today</div><div style='font-size:2.2rem; font-weight:800;'>{remaining}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Main layout
    left, right = st.columns([1.2, 1.0])

    with left:
        st.markdown('<div class="px-card">', unsafe_allow_html=True)
        st.markdown("## 🧪 Test Configuration")

        if st.button("🎯 Fill Sample Data", use_container_width=True):
            sample_fill()
            toast("Sample data filled ✅", icon="🎯")

        role = st.selectbox("Testing Role", TESTING_ROLES, index=0, key="role")
        test_type = st.selectbox("Test Type", TEST_TYPES, index=0, key="test_type")
        output_format = st.selectbox("Test Management Format (Import-Ready)", OUTPUT_FORMATS, index=0, key="output_format")

        context = st.text_area("Context (What are you testing?)", key="context", height=140)
        requirements = st.text_area("Requirements (What should it do?)", key="requirements", height=140)
        additional_info = st.text_area("Additional Info / Constraints (Optional)", key="additional_info", height=90)
        acceptance = st.text_area("Acceptance Criteria (Optional)", key="acceptance", height=90)
        edge_cases = st.text_area("Edge Cases (Optional)", key="edge_cases", height=90)
        nfr = st.text_area("Non-Functional Requirements (Optional)", key="nfr", height=90)

        st.markdown("### ✅ Coverage Options")
        coverage_flags = []
        cols = st.columns(2)
        for idx, flag in enumerate(COVERAGE_FLAGS):
            with cols[idx % 2]:
                if st.checkbox(flag, key=f"flag_{stable_id(flag)}"):
                    coverage_flags.append(flag)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="px-card">', unsafe_allow_html=True)
        st.markdown("## 🧩 Generated Prompt")

        generate = st.button("Generate Prompt", use_container_width=True)

        data = {
            "role": role,
            "test_type": test_type,
            "output_format": output_format,
            "context": context,
            "requirements": requirements,
            "additional_info": additional_info,
            "acceptance": acceptance,
            "edge_cases": edge_cases,
            "nfr": nfr,
            "coverage_flags": coverage_flags,
        }

        if generate:
            if not context.strip() or not requirements.strip():
                st.warning("Please fill at least **Context** and **Requirements**.")
            else:
                st.session_state["advanced_prompt"] = build_advanced_prompt(data)
                toast("Prompt generated ✅", icon="🧩")

        # Show prompt preview (kept)
        prompt_preview = st.session_state.get("advanced_prompt", "").strip()
        if prompt_preview:
            st.text_area("", value=prompt_preview, height=420, key="prompt_preview_box")
            clipboard_button("Copy Prompt", prompt_preview, btn_key="copy_prompt_preview")

        st.markdown("</div>", unsafe_allow_html=True)

    # Send to AI section (kept)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    send_ai = st.button("🚀 Send to AI", use_container_width=True)

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
        proceed = True

        # Free provider limit gate (decrement ONLY on Send to AI)
        if provider.key == "promptix_free":
            used_now = get_daily_used_count(user_id, today)  # refresh from DB
            remaining_now = max(0, DAILY_FREE_LIMIT - used_now)
            if remaining_now <= 0:
                # Popup + prevent the AI call
                toast("Free limits expired — 0 limits remaining today. Come back tomorrow (~24 hours).", icon="⏳")
                st.warning("Free limits expired. 0 limits remaining — come back tomorrow (~24 hours) or switch to BYOK in the sidebar.")
                proceed = False
            else:
                # keep UI counters consistent in this run
                used = used_now
                remaining = remaining_now

        # If we already hit the free limit, don't proceed further
        if not proceed:
            render_footer()
            return

        # Validate API key for BYOK
        if provider.needs_key and not api_key:
            st.error("Please add your API key in the sidebar for the selected provider (BYOK).")
        else:
            # Choose prompt
            prompt_to_send = st.session_state.get("advanced_prompt", "").strip()
            if not prompt_to_send:
                prompt_to_send = build_advanced_prompt(data)

            if not prompt_to_send.strip():
                st.warning("Please generate a prompt first (or fill required fields).")
            else:
                try:
                    with st.spinner("Calling AI..."):
                        if provider.vendor == "together":
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
                    used = new_used
                    remaining = new_remaining

                    # Update visible counters (sidebar + header) without changing UI
                    if limit_card_slot is not None:
                        limit_card_slot.markdown(f"""
                <div class="px-card" style="margin-top:10px;">
                  <div style="font-weight:700;">Free calls left today:</div>
                  <div style="font-size:1.25rem; margin-top:6px;">{remaining}/{DAILY_FREE_LIMIT}</div>
                  <div class="px-muted">Used today: {used}</div>
                </div>
                """, unsafe_allow_html=True)
                    if hero_remaining_slot is not None:
                        hero_remaining_slot.markdown(
                            f"<div class='px-muted'>Remaining today</div><div style='font-size:2.2rem; font-weight:800;'>{remaining}</div>",
                            unsafe_allow_html=True,
                        )

                    st.success(f"Done ✅ Free calls left today: {remaining}/{DAILY_FREE_LIMIT} (Used: {used})")
                    if remaining == 0:
                        # Popup after the 3rd free call
                        toast("Free limits expired — 0 limits remaining today. Come back tomorrow (~24 hours).", icon="⏳")
                        st.warning("Free limits expired. Come back tomorrow (~24 hours) or switch to BYOK in the sidebar.")

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
inject_css()

# 1) If a cookie operation is queued, execute it (ONE cookie call) and rerun
process_cookie_queue_once()

# 2) Bootstrap from cookie (ONE cookie read call)
bootstrap_session_from_cookie_once()

# 3) Route
render_main_app()
