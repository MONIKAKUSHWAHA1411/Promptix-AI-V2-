# app.py — Promptix AI (Version 2) — Supabase Login + Daily Limits (3/day) + Prompt Generator
#
# ✅ Works on Streamlit Cloud (no local setup needed)
# ✅ Fixes your issue: NO RuntimeError crash on login (shows real Supabase error in UI)
# ✅ Persists login via cookies + refresh token
# ✅ Persists daily free usage in Supabase table (so reload won’t reset 3/3)
#
# -----------------------------
# REQUIRED Streamlit Secrets (Streamlit Cloud → App → Settings → Secrets)
# -----------------------------
# SUPABASE_URL = "https://xxxx.supabase.co"
# SUPABASE_ANON_KEY = "your_anon_key"
#
# Optional (for default provider)
# DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
#
# -----------------------------
# REQUIRED pip packages (requirements.txt)
# -----------------------------
# streamlit
# requests
# extra-streamlit-components
#
# -----------------------------
# REQUIRED Supabase Table + RLS (run ONCE in Supabase → SQL Editor)
# -----------------------------
# create table if not exists public.promptix_usage (
#   id uuid primary key default gen_random_uuid(),
#   user_id uuid not null,
#   usage_date date not null,
#   count int not null default 0,
#   updated_at timestamptz not null default now(),
#   unique(user_id, usage_date)
# );
#
# alter table public.promptix_usage enable row level security;
#
# create policy "Users can read own usage"
# on public.promptix_usage for select
# using (auth.uid() = user_id);
#
# create policy "Users can insert own usage"
# on public.promptix_usage for insert
# with check (auth.uid() = user_id);
#
# create policy "Users can update own usage"
# on public.promptix_usage for update
# using (auth.uid() = user_id)
# with check (auth.uid() = user_id);

import json
import time
from datetime import datetime, date
from zoneinfo import ZoneInfo

import requests
import streamlit as st
import extra_streamlit_components as stx


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Promptix AI v2", layout="wide")

APP_TIMEZONE = st.secrets.get("APP_TIMEZONE", "Asia/Kolkata")  # daily reset timezone
DAILY_FREE_LIMIT = int(st.secrets.get("DAILY_FREE_LIMIT", 3))

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

DEFAULT_OPENAI_BASE_URL = st.secrets.get("DEFAULT_OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
DEFAULT_OPENAI_MODEL = st.secrets.get("DEFAULT_OPENAI_MODEL", "gpt-4o-mini").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing SUPABASE_URL / SUPABASE_ANON_KEY in Streamlit Secrets.")
    st.stop()

cookie_manager = stx.CookieManager()


# =========================================================
# UI STYLES
# =========================================================
st.markdown(
    """
<style>
/* background */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 700px at 20% 10%, rgba(255,0,90,0.10), transparent 60%),
              radial-gradient(1000px 600px at 80% 20%, rgba(0,180,255,0.10), transparent 60%),
              linear-gradient(180deg, #05070c 0%, #070a12 55%, #05070c 100%);
}

/* hide streamlit chrome */
header {visibility: hidden;}
footer {visibility: hidden;}

/* center auth card */
.center-auth{
  width: 100%;
  display:flex;
  justify-content:center;
  margin-top: 52px;
}
.px-card{
  width:min(860px, 94vw);
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 26px 26px 18px 26px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.55);
}
.px-h2{
  font-size: 26px;
  font-weight: 700;
  margin-bottom: 6px;
}
.px-muted{
  opacity: 0.75;
  margin-bottom: 16px;
}
.px-footer{
  margin-top: 36px;
  text-align:center;
  opacity:0.75;
  font-size: 13px;
}
.px-footer a{ color: #7dd3fc; text-decoration:none; }
.px-footer a:hover{ text-decoration:underline; }

/* sidebar links */
.px-links a { color:#7dd3fc !important; text-decoration:none; }
.px-links a:hover { text-decoration:underline; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def now_local_date() -> str:
    """Returns YYYY-MM-DD in APP_TIMEZONE."""
    tz = ZoneInfo(APP_TIMEZONE)
    return datetime.now(tz).date().isoformat()


def _safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return None


def _supabase_headers(access_token: str | None = None) -> dict:
    # Important: apikey is ALWAYS anon; Authorization should be user access token when reading/writing tables
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token or SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }


# =========================================================
# SUPABASE AUTH (NO CRASH)
# =========================================================
def supabase_sign_in(email: str, password: str):
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=password"
    try:
        r = requests.post(
            url,
            headers=_supabase_headers(None),
            json={"email": email, "password": password},
            timeout=20,
        )
    except Exception as e:
        return None, f"Network error: {e}"

    if r.status_code != 200:
        j = _safe_json(r)
        msg = None
        if isinstance(j, dict):
            msg = j.get("error_description") or j.get("message") or j.get("msg")
        if not msg:
            msg = r.text.strip()
        return None, f"{r.status_code}: {msg}"

    return r.json(), None


def supabase_sign_up(email: str, password: str):
    url = f"{SUPABASE_URL}/auth/v1/signup"
    try:
        r = requests.post(
            url,
            headers=_supabase_headers(None),
            json={"email": email, "password": password},
            timeout=20,
        )
    except Exception as e:
        return None, f"Network error: {e}"

    if r.status_code not in (200, 201):
        j = _safe_json(r)
        msg = None
        if isinstance(j, dict):
            msg = j.get("error_description") or j.get("message") or j.get("msg")
        if not msg:
            msg = r.text.strip()
        return None, f"{r.status_code}: {msg}"

    return r.json(), None


def supabase_get_user(access_token: str):
    url = f"{SUPABASE_URL}/auth/v1/user"
    try:
        r = requests.get(url, headers=_supabase_headers(access_token), timeout=15)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    return _safe_json(r)


def supabase_refresh(refresh_token: str):
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token"
    try:
        r = requests.post(
            url,
            headers=_supabase_headers(None),
            json={"refresh_token": refresh_token},
            timeout=20,
        )
    except Exception as e:
        return None, f"Network error: {e}"

    if r.status_code != 200:
        j = _safe_json(r)
        msg = None
        if isinstance(j, dict):
            msg = j.get("error_description") or j.get("message") or j.get("msg")
        if not msg:
            msg = r.text.strip()
        return None, f"{r.status_code}: {msg}"

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
    # store for 30 days
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

    # 1) Try validate access token
    user = supabase_get_user(access_token)
    if user and user.get("id") == user_id:
        st.session_state.user = {"id": user_id, "email": user.get("email", user_email or "")}
        st.session_state.access_token = access_token
        st.session_state.refresh_token = refresh_token
        return

    # 2) Refresh token if access token invalid/expired
    data, err = supabase_refresh(refresh_token)
    if err or not data:
        clear_cookie_session()
        return

    new_access = data.get("access_token", "")
    new_refresh = data.get("refresh_token", refresh_token)
    user_obj = data.get("user") or {}
    if not user_obj.get("id"):
        # fallback fetch
        user_obj = supabase_get_user(new_access) or {}

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
# SUPABASE USAGE TRACKING (PERSISTENT DAILY LIMIT)
# =========================================================
def get_daily_usage(user_id: str, access_token: str, usage_date: str) -> int:
    url = (
        f"{SUPABASE_URL}/rest/v1/promptix_usage"
        f"?user_id=eq.{user_id}&usage_date=eq.{usage_date}&select=count"
    )
    try:
        r = requests.get(url, headers=_supabase_headers(access_token), timeout=15)
    except Exception:
        return 0

    if r.status_code != 200:
        return 0

    j = _safe_json(r)
    if isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict):
        return int(j[0].get("count", 0) or 0)
    return 0


def set_daily_usage(user_id: str, access_token: str, usage_date: str, new_count: int) -> bool:
    # upsert by unique(user_id, usage_date)
    url = f"{SUPABASE_URL}/rest/v1/promptix_usage"
    headers = _supabase_headers(access_token)
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"

    payload = {"user_id": user_id, "usage_date": usage_date, "count": int(new_count), "updated_at": datetime.utcnow().isoformat()}

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
# LLM CALLS
# =========================================================
def call_openai_compatible(base_url: str, api_key: str, model: str, prompt: str) -> str:
    # Works for OpenAI + many “OpenAI-compatible” providers (OpenRouter, Groq-compatible, etc.)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a senior QA engineer. Generate high-quality test cases."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        j = _safe_json(r)
        raise RuntimeError(j.get("error", {}).get("message", r.text) if isinstance(j, dict) else r.text)
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4},
    }
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
    if r.status_code != 200:
        j = _safe_json(r)
        raise RuntimeError(j.get("error", {}).get("message", r.text) if isinstance(j, dict) else r.text)

    j = r.json()
    candidates = j.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
    return text.strip()


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
        "temperature": 0.4,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        j = _safe_json(r)
        raise RuntimeError(j.get("error", {}).get("message", r.text) if isinstance(j, dict) else r.text)

    j = r.json()
    content = j.get("content", [])
    if not content:
        return ""
    return content[0].get("text", "").strip()


def build_prompt(form: dict) -> str:
    return f"""
Create professional QA test cases for the following.

Product/Feature:
{form.get("feature","")}

User Story / Requirement:
{form.get("story","")}

Platform:
{form.get("platform","")}

Test Types Needed:
{form.get("types","")}

Acceptance Criteria:
{form.get("ac","")}

Edge Cases / Negative Scenarios to Include:
{form.get("neg","")}

Non-Functional Focus (if any):
{form.get("nfr","")}

Output Format (STRICT):
Return a markdown table with columns:
Test Case ID | Title | Preconditions | Steps | Expected Result | Priority | Type
Add 12–25 test cases. Include functional + negative + boundary + (if relevant) API/UI validations.
""".strip()


# =========================================================
# AUTH UI (CLEAN, NOT CONFUSING)
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
    st.markdown(
        '<div class="px-muted">Login is required to enforce accurate daily free limits.</div>',
        unsafe_allow_html=True,
    )

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
                        st.error("Login succeeded but session tokens are missing. Check Supabase auth settings.")
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
                    # Some projects require email confirmation => no session tokens returned
                    access_token = (data or {}).get("access_token", "")
                    refresh_token = (data or {}).get("refresh_token", "")
                    user_obj = (data or {}).get("user") or {}

                    if access_token and refresh_token and user_obj.get("id"):
                        apply_auth_session(user_obj, access_token, refresh_token)
                        st.success("Account created & logged in ✅")
                        st.rerun()
                    else:
                        st.success("Account created ✅")
                        st.info("If email confirmation is enabled in Supabase, please confirm your email, then login.")

    st.markdown("</div></div>", unsafe_allow_html=True)
    render_footer()


# =========================================================
# MAIN APP UI
# =========================================================
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")

    provider = st.sidebar.selectbox(
        "LLM Provider",
        ["OpenAI-compatible", "Gemini", "Anthropic"],
        index=0,
    )

    st.sidebar.markdown("### API Config")

    if provider == "OpenAI-compatible":
        base_url = st.sidebar.text_input("Base URL", value=DEFAULT_OPENAI_BASE_URL)
        model = st.sidebar.text_input("Model", value=DEFAULT_OPENAI_MODEL)
        api_key = st.sidebar.text_input("API Key", type="password", placeholder="Paste your key")
        st.session_state.llm_cfg = {"provider": provider, "base_url": base_url, "model": model, "api_key": api_key}

    elif provider == "Gemini":
        model = st.sidebar.text_input("Model", value="gemini-1.5-flash")
        api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Paste your key")
        st.session_state.llm_cfg = {"provider": provider, "model": model, "api_key": api_key}

    else:  # Anthropic
        model = st.sidebar.text_input("Model", value="claude-3-5-sonnet-latest")
        api_key = st.sidebar.text_input("Anthropic API Key", type="password", placeholder="Paste your key")
        st.session_state.llm_cfg = {"provider": provider, "model": model, "api_key": api_key}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Get your API keys")
    st.sidebar.markdown(
        """
<div class="px-links">
• <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI API keys</a><br/>
• <a href="https://aistudio.google.com/app/apikey" target="_blank">Gemini API key</a><br/>
• <a href="https://console.anthropic.com/" target="_blank">Anthropic API key</a><br/>
</div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        do_logout()


def generate_with_selected_provider(prompt: str) -> str:
    cfg = st.session_state.get("llm_cfg") or {}
    provider = cfg.get("provider")

    if provider == "OpenAI-compatible":
        if not cfg.get("api_key"):
            raise RuntimeError("Missing API key.")
        return call_openai_compatible(cfg.get("base_url", DEFAULT_OPENAI_BASE_URL), cfg["api_key"], cfg.get("model", DEFAULT_OPENAI_MODEL), prompt)

    if provider == "Gemini":
        if not cfg.get("api_key"):
            raise RuntimeError("Missing Gemini API key.")
        return call_gemini(cfg["api_key"], cfg.get("model", "gemini-1.5-flash"), prompt)

    if provider == "Anthropic":
        if not cfg.get("api_key"):
            raise RuntimeError("Missing Anthropic API key.")
        return call_anthropic(cfg["api_key"], cfg.get("model", "claude-3-5-sonnet-latest"), prompt)

    raise RuntimeError("Invalid provider.")


def render_main_app():
    render_sidebar()

    user = st.session_state.user
    access_token = st.session_state.access_token
    usage_date = now_local_date()

    used = get_daily_usage(user["id"], access_token, usage_date)
    remaining = max(0, DAILY_FREE_LIMIT - used)

    st.markdown("# Promptix AI v2")
    st.caption("Generate QA test cases with consistent daily free limits (reload-safe).")

    colA, colB, colC = st.columns([1.2, 1, 1])
    with colA:
        st.info(f"👤 Logged in as **{user.get('email','')}**")
    with colB:
        st.metric("Daily free limit", f"{DAILY_FREE_LIMIT}/day")
    with colC:
        st.metric("Remaining today", remaining)

    st.progress((used / DAILY_FREE_LIMIT) if DAILY_FREE_LIMIT > 0 else 0)

    st.markdown("---")

    with st.form("prompt_form"):
        feature = st.text_input("Feature / Module", placeholder="e.g., Login + OTP, Payments, Search, Checkout")
        story = st.text_area("User story / requirement", height=120, placeholder="Paste requirement / story / spec...")
        platform = st.selectbox("Platform", ["Web", "Mobile", "API", "Web + API", "Mobile + API"], index=0)
        types = st.multiselect(
            "Test types",
            ["Functional", "Regression", "Smoke", "Sanity", "UI", "API", "Integration", "E2E", "Accessibility", "Security (basic)", "Performance (basic)"],
            default=["Functional", "UI"],
        )
        ac = st.text_area("Acceptance criteria (optional)", height=90, placeholder="List acceptance criteria, bullet points...")
        neg = st.text_area("Edge cases / negative scenarios (optional)", height=90, placeholder="Invalid inputs, boundary values, error states...")
        nfr = st.text_input("Non-functional focus (optional)", placeholder="e.g., latency < 200ms, WCAG, OWASP basics")

        submit = st.form_submit_button("Generate Test Cases", use_container_width=True)

    if submit:
        if remaining <= 0:
            st.error("Daily free limit reached. Please come back tomorrow.")
            render_footer()
            return

        if not feature or not story:
            st.error("Please provide Feature and User story/requirement.")
            render_footer()
            return

        prompt = build_prompt(
            {
                "feature": feature,
                "story": story,
                "platform": platform,
                "types": ", ".join(types) if types else "",
                "ac": ac,
                "neg": neg,
                "nfr": nfr,
            }
        )

        with st.spinner("Generating…"):
            try:
                output = generate_with_selected_provider(prompt)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                render_footer()
                return

        # only count usage if generation succeeded
        new_used = increment_daily_usage(user["id"], access_token, usage_date)

        st.success(f"Done ✅  (Used today: {new_used}/{DAILY_FREE_LIMIT})")
        st.markdown(output)

        st.download_button(
            "Download as .md",
            data=output.encode("utf-8"),
            file_name=f"promptix_testcases_{usage_date}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    render_footer()


# =========================================================
# APP ENTRY
# =========================================================
try_bootstrap_session_from_cookie()

if not is_authed():
    render_login()
    st.stop()

render_main_app()
