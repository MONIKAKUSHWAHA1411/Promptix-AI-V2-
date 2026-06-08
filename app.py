import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "Promptix AI v2"
DAILY_FREE_LIMIT = 3
LOCAL_TZ = "Asia/Kolkata"  # used only for display messaging (date reset is per local machine date)

# Together defaults (Promptix Free)
TOGETHER_DEFAULT_ENDPOINT = "https://api.together.xyz/v1/chat/completions"
TOGETHER_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# FIX #4: Supabase credentials moved to lazy-load helper — NOT read at module level
# (avoids calling st.secrets before st.set_page_config)


def _get_supabase_config() -> Tuple[str, str]:
    """Lazy-load Supabase config after page setup."""
    url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", "")).strip()
    key = st.secrets.get("SUPABASE_ANON_KEY", os.getenv("SUPABASE_ANON_KEY", "")).strip()
    return url, key


# =========================================================
# PAGE SETUP + STYLES (keep same dark UI vibe)
# =========================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")

GLOBAL_CSS = """
<style>
/* Hide default Streamlit chrome spacing a bit */
header[data-testid="stHeader"] {visibility: hidden; height: 0px;}
div.block-container {padding-top: 1.2rem;}

/* App background (dark gradient) */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 20% 10%, rgba(120,40,255,0.25), transparent 60%),
              radial-gradient(1200px 600px at 80% 20%, rgba(0,200,255,0.18), transparent 60%),
              linear-gradient(180deg, #070A12 0%, #060914 100%);
  color: #EAEAF2;
}
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border-right: 1px solid rgba(255,255,255,0.06);
}

h1, h2, h3, h4, h5, h6, p, div, span, label {color: #EAEAF2;}
a {color: #63B3FF !important; text-decoration: none;}
a:hover {text-decoration: underline;}

.card {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.04);
  border-radius: 16px;
  padding: 18px 18px;
}
.hero {
  background: linear-gradient(90deg, rgba(120,40,255,0.55), rgba(80,50,255,0.30));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 18px;
}
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  margin-left: 8px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.12);
}
.small-muted {opacity: 0.75; font-size: 13px;}
.hrline {
  height: 14px;
  width: 72%;
  margin: 8px auto 22px auto;
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.07);
}
.kpiBox {
  border-radius: 12px;
  padding: 14px 14px;
  background: rgba(80,140,255,0.14);
  border: 1px solid rgba(80,140,255,0.22);
}
.kpiGood {
  border-radius: 12px;
  padding: 14px 14px;
  background: rgba(60,220,140,0.14);
  border: 1px solid rgba(60,220,140,0.22);
}
.btnRow button {
  width: 100%;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# =========================================================
# IN-MEMORY DAILY USAGE STORE (fixes reload reset)
# =========================================================
@st.cache_resource
def _usage_store() -> Dict[str, Dict[str, int]]:
    # { user_key: { "YYYY-MM-DD": used_count } }
    return {}


def _today_str() -> str:
    # Using server local date. (For Streamlit Cloud it is stable.)
    return datetime.now().strftime("%Y-%m-%d")


def _user_key_from_email(email: str) -> str:
    # stable + anonymous-ish key
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()


def get_used_today(user_key: str) -> int:
    store = _usage_store()
    d = _today_str()
    return int(store.get(user_key, {}).get(d, 0))


def set_used_today(user_key: str, used: int) -> None:
    store = _usage_store()
    d = _today_str()
    if user_key not in store:
        store[user_key] = {}
    store[user_key][d] = int(max(0, used))


def inc_used_today(user_key: str) -> int:
    used = get_used_today(user_key)
    used += 1
    set_used_today(user_key, used)
    return used


# =========================================================
# AUTH (Supabase optional)
# =========================================================
def supabase_enabled() -> bool:
    url, key = _get_supabase_config()
    return bool(url and key)


def _sb_headers(auth_bearer: Optional[str] = None) -> Dict[str, str]:
    _, anon_key = _get_supabase_config()
    headers = {
        "apikey": anon_key,
        "Content-Type": "application/json",
    }
    if auth_bearer:
        headers["Authorization"] = f"Bearer {auth_bearer}"
    return headers


def supabase_signup(email: str, password: str) -> Tuple[bool, str]:
    supabase_url, _ = _get_supabase_config()
    try:
        url = f"{supabase_url}/auth/v1/signup"
        r = requests.post(url, headers=_sb_headers(), json={"email": email, "password": password}, timeout=20)
        if r.status_code in (200, 201):
            return True, "Account created. Please login."
        return False, f"Signup failed: {r.text}"
    except Exception as e:
        return False, f"Signup error: {e}"


def supabase_login(email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    supabase_url, _ = _get_supabase_config()
    try:
        url = f"{supabase_url}/auth/v1/token?grant_type=password"
        r = requests.post(url, headers=_sb_headers(), json={"email": email, "password": password}, timeout=20)
        if r.status_code == 200:
            data = r.json()
            # data: access_token, user, etc.
            return True, "Login successful.", data
        return False, f"Login failed: {r.text}", None
    except Exception as e:
        return False, f"Login error: {e}", None


def init_session():
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_email", "")
    st.session_state.setdefault("user_id", "")  # from Supabase if available
    st.session_state.setdefault("access_token", "")

    # UI state
    st.session_state.setdefault("provider", "Promptix Free (LLAMA via Together)")
    st.session_state.setdefault("api_endpoint", TOGETHER_DEFAULT_ENDPOINT)
    st.session_state.setdefault("model", TOGETHER_DEFAULT_MODEL)

    st.session_state.setdefault("user_api_key", "")
    st.session_state.setdefault("generated_prompt", "")
    st.session_state.setdefault("ai_response", "")

    st.session_state.setdefault("testing_role", "Automation Tester – Test automation specialist")
    st.session_state.setdefault("test_type", "Regression Testing")
    st.session_state.setdefault("format_type", "Cucumber/BDD – Gherkin syntax")
    st.session_state.setdefault("context_text", "")
    st.session_state.setdefault("req_text", "")

    st.session_state.setdefault("_show_limit_dialog", False)

    # FIX #7: per-session generation history (last 5)
    st.session_state.setdefault("generation_history", [])


init_session()


# =========================================================
# LIMIT POPUP (blocks 4th call)
# =========================================================
def open_limit_dialog():
    st.session_state["_show_limit_dialog"] = True


def render_limit_dialog():
    if not st.session_state.get("_show_limit_dialog"):
        return

    msg = (
        f"Your free daily limit (**{DAILY_FREE_LIMIT}**) is exhausted.\n\n"
        f"✅ Please come back tomorrow (resets at midnight, {LOCAL_TZ}).\n\n"
        f"Tip: Use your own API key (OpenAI/Gemini/Anthropic/Together) to continue without the free limit."
    )

    # Try Streamlit dialog (popup)
    try:
        @st.dialog("Free limits expired")
        def _dlg():
            st.write(msg)
            if st.button("OK", key="limit_dialog_ok_btn"):
                st.session_state["_show_limit_dialog"] = False
                st.rerun()
        _dlg()
    except Exception:
        # Fallback if st.dialog not available in deployed Streamlit
        st.warning(msg)
        if st.button("Dismiss", key="limit_dialog_dismiss_btn"):
            st.session_state["_show_limit_dialog"] = False
            st.rerun()


# =========================================================
# LLM CALLS (keeps existing behavior: "Send to AI")
# FIX #5: max_tokens increased from 1400 to 2500 in all calls
# =========================================================
def call_chat_completions(endpoint: str, api_key: str, model: str, messages, temperature=0.2, max_tokens=2500) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # OpenAI-style
    return data["choices"][0]["message"]["content"]


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    # FIX #1: API key passed as x-goog-api-key header instead of URL query param
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_anthropic(api_key: str, model: str, system: str, user_prompt: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2500,  # FIX #5: increased from 1400
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["content"][0]["text"]


# =========================================================
# PROMPT BUILDING (keeps existing functionality)
# =========================================================
ROLES = [
    "Automation Tester – Test automation specialist",
    "Manual QA – Functional tester",
    "QA Lead – Test strategy & planning",
    "SDET – Framework + CI/CD quality",
    "Performance QA – Load & reliability",
    "Security QA – AppSec validation",
]

TEST_TYPES = [
    "Regression Testing",
    "Smoke Testing",
    "Sanity Testing",
    "Functional Testing",
    "Integration Testing",
    "API Testing",
    "UI Testing",
    "Performance Testing",
    "Security Testing",
    "Accessibility Testing",
]

FORMATS = [
    "Cucumber/BDD – Gherkin syntax",
    "Jira – Test case format",
    "TestRail – Steps/Expected format",
    "CSV – Spreadsheet style",
    "Plain – Structured bullets",
]

SAMPLE_CONTEXT = "zepto.com — user added a new address"
SAMPLE_REQ = """1) User can add a new delivery address with name, phone, pincode, city, state, landmark.
2) Validate pincode: only 6 digits.
3) Phone: 10 digits, numeric only.
4) Save address and show it in address list.
5) If user selects default address, it should be used at checkout.
6) Handle network failure gracefully with retry.
"""


def build_prompt(role: str, test_type: str, format_type: str, context: str, reqs: str) -> str:
    return f"""
You are a senior QA engineer.

Goal:
Generate comprehensive, export-ready test cases for the scenario below.

Testing Role: {role}
Test Type: {test_type}
Output Format: {format_type}

Context (What are you testing?):
{context.strip()}

Requirements (What should it do?):
{reqs.strip()}

Instructions:
- Include positive, negative, boundary, and edge cases.
- Include validation tests, error handling, and UX checks where relevant.
- Add preconditions, test data, steps, expected results.
- Add a small section for "Non-functional considerations" if relevant (performance/security/accessibility).
- Keep output clean and import-ready for the selected format.
""".strip()


# =========================================================
# FIX #6: Smart download filename based on format
# =========================================================
def _download_filename(format_type: str) -> str:
    fmt = format_type.lower()
    if "bdd" in fmt or "gherkin" in fmt or "cucumber" in fmt:
        return "test_cases.feature"
    elif "csv" in fmt:
        return "test_cases.csv"
    else:
        return "test_cases.txt"


# =========================================================
# FIX #7: Generation history helpers
# =========================================================
def _add_to_history(context: str, format_type: str, ai_response: str) -> None:
    """Prepend a generation record; keep only the last 5."""
    history = st.session_state.get("generation_history", [])
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "context_preview": (context or "").strip()[:80],
        "format_type": format_type,
        "ai_response": ai_response,
    }
    history.insert(0, record)
    st.session_state["generation_history"] = history[:5]


# =========================================================
# LOGIN PAGE (fixes DuplicateElementKey by unique keys)
# =========================================================
def render_login_page():
    # Hide sidebar on login screen to match your current UI experience
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;}</style>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hrline"></div>', unsafe_allow_html=True)
    st.markdown("## 🔐 Login to Promptix")
    st.markdown('<div class="small-muted">Login is required to enforce accurate daily free limits.</div>', unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Login", "Create account"])

    with tab_login:
        email = st.text_input("Email", value=st.session_state.get("user_email", ""), key="login_email_input")
        pw = st.text_input("Password", type="password", key="login_password_input")

        if st.button("Login", key="login_btn", use_container_width=True):
            email = (email or "").strip()
            if not email or not pw:
                st.error("Please enter email and password.")
                st.stop()

            if supabase_enabled():
                ok, msg, data = supabase_login(email, pw)
                if not ok:
                    st.error(msg)
                    st.stop()

                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.access_token = data.get("access_token", "")
                user = data.get("user") or {}
                st.session_state.user_id = user.get("id", "")

                st.rerun()
            else:
                # Local fallback auth (demo)
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.user_id = ""
                st.session_state.access_token = ""
                st.rerun()

    with tab_signup:
        email2 = st.text_input("Email", key="signup_email_input")
        pw2 = st.text_input("Password", type="password", key="signup_password_input")
        if st.button("Create account", key="create_account_btn", use_container_width=True):
            email2 = (email2 or "").strip()
            if not email2 or not pw2:
                st.error("Please enter email and password.")
                st.stop()

            if not supabase_enabled():
                st.error("Supabase is not configured (SUPABASE_URL / SUPABASE_ANON_KEY missing).")
                st.stop()

            ok, msg = supabase_signup(email2, pw2)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    # IMPORTANT: stop here so main app widgets are not created in same run (prevents 1-second errors)
    st.stop()


# =========================================================
# SIDEBAR (keeps same structure, and fixes free limit display)
# =========================================================
def render_sidebar(user_key: str):
    st.sidebar.markdown("")

    provider = st.sidebar.selectbox(
        "Provider",
        [
            "Promptix Free (LLAMA via Together)",
            "OpenAI (Your key)",
            "Gemini (Your key)",
            "Anthropic (Your key)",
            "Together (Your key)",
        ],
        key="provider_selectbox",
    )
    st.session_state.provider = provider

    # Defaults by provider
    if provider == "Promptix Free (LLAMA via Together)":
        st.session_state.api_endpoint = TOGETHER_DEFAULT_ENDPOINT
        st.session_state.model = TOGETHER_DEFAULT_MODEL
    elif provider == "OpenAI (Your key)":
        st.session_state.api_endpoint = "https://api.openai.com/v1/chat/completions"
        st.session_state.model = "gpt-4o-mini"
    elif provider == "Gemini (Your key)":
        st.session_state.api_endpoint = "(Gemini REST — auto)"
        st.session_state.model = "gemini-1.5-flash"
    elif provider == "Anthropic (Your key)":
        st.session_state.api_endpoint = "https://api.anthropic.com/v1/messages"
        st.session_state.model = "claude-3-5-sonnet-latest"
    elif provider == "Together (Your key)":
        st.session_state.api_endpoint = TOGETHER_DEFAULT_ENDPOINT
        st.session_state.model = TOGETHER_DEFAULT_MODEL

    st.sidebar.text_input("API Endpoint", value=st.session_state.api_endpoint, key="api_endpoint_input")
    st.sidebar.text_input("Model", value=st.session_state.model, key="model_input")

    st.sidebar.markdown("🔗 [Get your API key](https://platform.openai.com/api-keys)")
    st.sidebar.markdown("- OpenAI: [Get your API key](https://platform.openai.com/api-keys)")
    st.sidebar.markdown("- Gemini: [Get your API key](https://aistudio.google.com/app/apikey)")
    st.sidebar.markdown("- Anthropic: [Get your API key](https://console.anthropic.com/settings/keys)")
    st.sidebar.markdown("- Together: [Get your API key](https://api.together.xyz/settings/api-keys)")

    # FREE LIMIT BOX (FIXED: now computed from stored counter)
    used = get_used_today(user_key)
    remaining = max(0, DAILY_FREE_LIMIT - used)

    st.sidebar.markdown(
        f"""
        <div class="kpiBox">
          <div style="font-weight:700;">Free calls left today (this session):</div>
          <div style="font-size:20px; font-weight:800; margin-top:6px;">{remaining}/{DAILY_FREE_LIMIT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show which key is being used
    if provider == "Promptix Free (LLAMA via Together)":
        st.sidebar.markdown(
            """
            <div class="kpiGood" style="margin-top:12px;">
              <div style="font-weight:700;">Using server key from env:</div>
              <div style="font-weight:800; margin-top:6px;">TOGETHER_API_KEY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.session_state.user_api_key = st.sidebar.text_input("Your API Key", type="password", key="user_api_key_input")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.button("💾 Save", key="save_btn", use_container_width=True)
    with c2:
        if st.button("🧹 Clear", key="clear_btn", use_container_width=True):
            st.session_state.context_text = ""
            st.session_state.req_text = ""
            st.session_state.generated_prompt = ""
            st.session_state.ai_response = ""
            st.rerun()

    st.sidebar.markdown("---")

    # FIX #3: removed dead st.stop() after st.rerun() — st.rerun() raises RerunException
    # so st.stop() was unreachable
    if st.sidebar.button("Logout", key="logout_btn", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.user_id = ""
        st.session_state.access_token = ""
        st.rerun()

    st.sidebar.caption("Security: never hardcode or commit keys to GitHub.")


# =========================================================
# MAIN APP UI (keeps same UX, only fixes limits + auth error)
# =========================================================
def render_main_app():
    email = st.session_state.user_email.strip()
    if not email:
        # Should not happen if login gate works, but keep safe.
        render_login_page()

    user_key = _user_key_from_email(email)

    render_sidebar(user_key)

    # Header
    st.markdown(
        f"""
        <div class="hero">
          <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:28px; font-weight:900;">{APP_TITLE}</div>
            <span class="badge">MVP</span>
          </div>
          <div class="small-muted" style="margin-top:6px;">
            Turn product requirements into structured, export-ready test cases — with edge cases, negatives, and multiple formats (Jira/TestRail/BDD/CSV).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hrline"></div>', unsafe_allow_html=True)

    # FIX: show the popup if needed
    render_limit_dialog()

    st.markdown("## 🧪 Test Configuration")

    # Sample data
    if st.button("🎯 Fill Sample Data", key="fill_sample_data_btn", use_container_width=False):
        st.session_state.context_text = SAMPLE_CONTEXT
        st.session_state.req_text = SAMPLE_REQ
        st.rerun()

    st.session_state.testing_role = st.selectbox("Testing Role", ROLES, key="testing_role_select")
    st.session_state.test_type = st.selectbox("Test Type", TEST_TYPES, key="test_type_select")
    st.session_state.format_type = st.selectbox("Test Management Format (Import-Ready)", FORMATS, key="format_select")

    st.session_state.context_text = st.text_area(
        "Context (What are you testing?)",
        value=st.session_state.context_text,
        height=120,
        key="context_textarea",
    )

    st.session_state.req_text = st.text_area(
        "Requirements (What should it do?)",
        value=st.session_state.req_text,
        height=180,
        key="req_textarea",
    )

    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        st.markdown("### 🧩 Generated Prompt")
        if st.button("Generate Prompt", key="generate_prompt_btn", use_container_width=True):
            st.session_state.generated_prompt = build_prompt(
                st.session_state.testing_role,
                st.session_state.test_type,
                st.session_state.format_type,
                st.session_state.context_text,
                st.session_state.req_text,
            )
            st.rerun()

        st.session_state.generated_prompt = st.text_area(
            "",
            value=st.session_state.generated_prompt,
            height=260,
            key="generated_prompt_textarea",
            label_visibility="collapsed",
        )

    with colB:
        st.markdown("### 🤖 AI Response")
        # FIX #9: editable text area for the response (kept for editing)
        st.session_state.ai_response = st.text_area(
            "",
            value=st.session_state.ai_response,
            height=200,
            key="ai_response_textarea",
            label_visibility="collapsed",
        )

        # FIX #9: also render the AI response in a st.code block for easy copy
        if st.session_state.ai_response:
            st.code(st.session_state.ai_response, language="markdown")

        # FIX #6: download button with smart filename
        if st.session_state.ai_response:
            filename = _download_filename(st.session_state.format_type)
            st.download_button(
                label=f"⬇️ Download ({filename})",
                data=st.session_state.ai_response,
                file_name=filename,
                mime="text/plain",
                key="download_ai_response_btn",
            )

    st.markdown("---")

    # SEND TO AI (THIS IS WHERE LIMIT FIX IS APPLIED)
    provider = st.session_state.provider
    used = get_used_today(user_key)
    remaining = max(0, DAILY_FREE_LIMIT - used)

    send_col1, send_col2 = st.columns([2, 3], gap="large")
    with send_col1:
        st.markdown("#### Send to AI")
        st.caption("Uses 1 free call only on Promptix Free provider.")

        send_clicked = st.button("Send to AI", key="send_to_ai_btn", use_container_width=True)

    with send_col2:
        st.markdown("#### Daily free limit")
        st.write(f"**{DAILY_FREE_LIMIT}/day**  •  **Remaining today: {remaining}**")

    if send_clicked:
        # 1) block if free provider & no remaining
        if provider == "Promptix Free (LLAMA via Together)":
            if remaining <= 0:
                open_limit_dialog()
                st.toast("Free limits expired. Come back tomorrow.", icon="⏳")
                st.rerun()

        # 2) require prompt
        prompt = (st.session_state.generated_prompt or "").strip()
        if not prompt:
            st.error("Please click **Generate Prompt** first (or paste your prompt in the prompt box).")
            st.stop()

        # 3) choose provider key + call
        try:
            with st.spinner("Calling AI..."):
                if provider == "Promptix Free (LLAMA via Together)":
                    api_key = st.secrets.get("TOGETHER_API_KEY", os.getenv("TOGETHER_API_KEY", "")).strip()
                    if not api_key:
                        st.error("TOGETHER_API_KEY is not set in Streamlit Secrets / environment.")
                        st.stop()

                    out = call_chat_completions(
                        endpoint=TOGETHER_DEFAULT_ENDPOINT,
                        api_key=api_key,
                        model=TOGETHER_DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a senior QA engineer. Output must be clean and import-ready."},
                            {"role": "user", "content": prompt},
                        ],
                    )

                    # Increment counter after successful call
                    inc_used_today(user_key)

                elif provider == "Together (Your key)":
                    api_key = (st.session_state.user_api_key or "").strip()
                    if not api_key:
                        st.error("Please enter your Together API key in the sidebar.")
                        st.stop()

                    out = call_chat_completions(
                        endpoint=TOGETHER_DEFAULT_ENDPOINT,
                        api_key=api_key,
                        model=st.session_state.model_input if "model_input" in st.session_state else TOGETHER_DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a senior QA engineer. Output must be clean and import-ready."},
                            {"role": "user", "content": prompt},
                        ],
                    )

                elif provider == "OpenAI (Your key)":
                    api_key = (st.session_state.user_api_key or "").strip()
                    if not api_key:
                        st.error("Please enter your OpenAI API key in the sidebar.")
                        st.stop()

                    out = call_chat_completions(
                        endpoint="https://api.openai.com/v1/chat/completions",
                        api_key=api_key,
                        model=st.session_state.model_input if "model_input" in st.session_state else "gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a senior QA engineer. Output must be clean and import-ready."},
                            {"role": "user", "content": prompt},
                        ],
                    )

                elif provider == "Gemini (Your key)":
                    api_key = (st.session_state.user_api_key or "").strip()
                    if not api_key:
                        st.error("Please enter your Gemini API key in the sidebar.")
                        st.stop()

                    model = st.session_state.model_input if "model_input" in st.session_state else "gemini-1.5-flash"
                    out = call_gemini(api_key=api_key, model=model, prompt=prompt)

                elif provider == "Anthropic (Your key)":
                    api_key = (st.session_state.user_api_key or "").strip()
                    if not api_key:
                        st.error("Please enter your Anthropic API key in the sidebar.")
                        st.stop()

                    model = st.session_state.model_input if "model_input" in st.session_state else "claude-3-5-sonnet-latest"
                    out = call_anthropic(
                        api_key=api_key,
                        model=model,
                        system="You are a senior QA engineer. Output must be clean and import-ready.",
                        user_prompt=prompt,
                    )
                else:
                    st.error("Unknown provider.")
                    st.stop()

            st.session_state.ai_response = out

            # FIX #7: save to generation history
            _add_to_history(
                context=st.session_state.context_text,
                format_type=st.session_state.format_type,
                ai_response=out,
            )

            # If limit just reached, show popup next
            if provider == "Promptix Free (LLAMA via Together)":
                used_now = get_used_today(user_key)
                if used_now >= DAILY_FREE_LIMIT:
                    st.toast("Free limit reached for today.", icon="✅")

            st.rerun()

        except requests.HTTPError as e:
            # FIX #8: human-readable error with actionable tip
            st.error(
                f"AI call failed (HTTP {e.response.status_code if e.response is not None else 'unknown'}). "
                f"Please check your API key or try a different provider."
            )
        except Exception as e:
            # FIX #8: human-readable error with actionable tip
            st.error(
                f"AI call failed: {e}\n\n"
                f"Tip: Check your API key or try a different provider."
            )

    # =========================================================
    # FIX #7: Generation History expander
    # =========================================================
    history = st.session_state.get("generation_history", [])
    if history:
        with st.expander("📜 History (last 5 generations)", expanded=False):
            for i, record in enumerate(history):
                st.markdown(
                    f"**{record['timestamp']}** — Format: `{record['format_type']}`  "
                    f"Context: _{record['context_preview'] or '(empty)'}{'...' if len(record.get('context_preview','')) == 80 else ''}_"
                )
                if st.button(f"Re-load #{i + 1}", key=f"history_reload_{i}"):
                    st.session_state.ai_response = record["ai_response"]
                    st.rerun()
                st.markdown("---")


# =========================================================
# ROUTER (auth gate) — FIXES the 1-second error by stopping
# =========================================================
if not st.session_state.logged_in:
    render_login_page()
else:
    render_main_app()
