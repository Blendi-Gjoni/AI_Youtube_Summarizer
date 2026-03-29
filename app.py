import html
import json
import os
import time
import hashlib
import streamlit as st
from dotenv import load_dotenv
from chunking import fetch_transcript, transcript_to_documents, chunk_docs
from summarize import summarize_chunks, combine_summaries, generate_quiz, answer_question

load_dotenv()

st.set_page_config(page_title="YT Summarizer", page_icon="🎬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"]{background-color:#0a0a0a;color:#f0ede6;font-family:'DM Mono',monospace;}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse 80% 50% at 50% -10%,rgba(255,75,43,0.12) 0%,transparent 70%),#0a0a0a;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"]{display:none;}
[data-testid="stSidebar"]{display:none;}
.block-container{max-width:720px;padding:3rem 2rem 4rem;margin:0 auto;}
[data-testid="stTextInput"] input{background:#111 !important;border:1px solid rgba(255,255,255,0.1) !important;border-radius:8px !important;color:#f0ede6 !important;font-family:'DM Mono',monospace !important;font-size:0.85rem !important;padding:0.75rem 1rem !important;}
[data-testid="stTextInput"] input:focus{border-color:#ff4b2b !important;box-shadow:0 0 0 2px rgba(255,75,43,0.15) !important;}
[data-testid="stTextInput"] label{color:#6b6760 !important;font-size:0.72rem !important;letter-spacing:0.1em !important;text-transform:uppercase !important;font-family:'DM Mono',monospace !important;}
[data-testid="stButton"] button{width:100%;background:#ff4b2b !important;color:#0a0a0a !important;border:none !important;border-radius:8px !important;font-family:'Syne',sans-serif !important;font-weight:700 !important;font-size:0.85rem !important;letter-spacing:0.08em !important;text-transform:uppercase !important;padding:0.75rem 1.5rem !important;margin-top:0.75rem !important;}
[data-testid="stButton"] button:hover{opacity:0.88 !important;}
[data-testid="stDownloadButton"] button{width:100%;background:transparent !important;color:#f0ede6 !important;border:1px solid rgba(255,255,255,0.15) !important;border-radius:8px !important;font-family:'Syne',sans-serif !important;font-weight:600 !important;font-size:0.8rem !important;letter-spacing:0.08em !important;text-transform:uppercase !important;padding:0.65rem 1.5rem !important;margin-top:0.5rem !important;}
[data-testid="stDownloadButton"] button:hover{border-color:#ff4b2b !important;color:#ff4b2b !important;}
[data-testid="stAlert"]{background:rgba(255,75,43,0.08) !important;border:1px solid rgba(255,75,43,0.3) !important;border-radius:8px !important;color:#ff4b2b !important;font-family:'DM Mono',monospace !important;font-size:0.8rem !important;}
[data-testid="stRadio"] label{color:#c8c4bc !important;font-size:0.88rem !important;font-family:'DM Mono',monospace !important;}
[data-testid="stRadio"] div[role="radiogroup"]{gap:0.5rem;}
[data-testid="stProgress"] > div > div{background:#ff4b2b !important;border-radius:4px;}
[data-testid="stProgress"]{background:rgba(255,255,255,0.07) !important;border-radius:4px;}
.footer{text-align:center;margin-top:3rem;font-size:0.7rem;color:#6b6760 !important;letter-spacing:0.1em;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DAILY_SUMMARIZE_LIMIT = 5
QA_LIMIT = 3
RATE_LIMIT_DIR = "/tmp/yt_rate_limits"
os.makedirs(RATE_LIMIT_DIR, exist_ok=True)

# ── Session state init ─────────────────────────────────────────────────────────
defaults = {
    "sections": None,
    "summaries": None,
    "quiz": None,
    "quiz_index": 0,
    "quiz_answers": [],
    "quiz_answered_current": False,
    "quiz_selected": None,
    "quiz_done": False,
    "qa_answer": None,
    "qa_count": 0,  # questions asked for current video
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Rate limiting ──────────────────────────────────────────────────────────────

def get_ip() -> str:
    """Get a hashed identifier for the current user."""
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", headers.get("X-Real-IP", "unknown"))
        # take first IP if there's a chain
        ip = ip.split(",")[0].strip()
    except Exception:
        ip = "unknown"
    return hashlib.md5(ip.encode()).hexdigest()


def get_rate_limit_file(user_id: str) -> str:
    today = time.strftime("%Y-%m-%d")
    return os.path.join(RATE_LIMIT_DIR, f"{user_id}_{today}.json")


def get_usage(user_id: str) -> dict:
    path = get_rate_limit_file(user_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"summarize_count": 0}


def increment_usage(user_id: str):
    path = get_rate_limit_file(user_id)
    usage = get_usage(user_id)
    usage["summarize_count"] += 1
    with open(path, "w") as f:
        json.dump(usage, f)


def is_rate_limited(user_id: str) -> bool:
    return get_usage(user_id)["summarize_count"] >= DAILY_SUMMARIZE_LIMIT


def summarize_count(user_id: str) -> int:
    return get_usage(user_id)["summarize_count"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def seconds_to_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_summary(raw: str) -> dict:
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        data = json.loads(cleaned)

        def to_bullets(value) -> str:
            if isinstance(value, list):
                return "\n".join(f"* {item}" for item in value if item)
            return str(value).strip()

        return {
            "tldr": str(data.get("tldr", "")).strip(),
            "key_points": to_bullets(data.get("key_points", [])),
            "action_items": to_bullets(data.get("action_items", [])),
        }
    except Exception:
        return {"tldr": raw.strip(), "key_points": "", "action_items": ""}


def content_to_html(content: str) -> str:
    lines = content.strip().splitlines()
    html_lines = []
    in_list = False
    for line in lines:
        stripped = html.escape(line.strip().lstrip("-•*").strip())
        if line.strip().startswith(("-", "•", "*")):
            if not in_list:
                html_lines.append("<ul style='margin:0;padding-left:1.2rem;'>")
                in_list = True
            html_lines.append(f"<li style='margin-bottom:0.4rem;'>{stripped}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if stripped:
                html_lines.append(f"<p style='margin:0 0 0.4rem;'>{stripped}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def card(label: str, body_html: str):
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);"
        f"border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem;'>"
        f"<div style='font-family:DM Mono,monospace;font-size:0.65rem;font-weight:500;"
        f"letter-spacing:0.2em;text-transform:uppercase;color:#ff4b2b;margin-bottom:0.75rem;'>{label}</div>"
        f"<div style='font-size:0.88rem;line-height:1.75;color:#c8c4bc;'>{body_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_section(label: str, content: str):
    if not content.strip():
        return
    card(label, content_to_html(content))


def render_chapters(summaries: list):
    skip_prefixes = ("key point", "action item", "tl;dr", "tldr")
    with st.expander("📑 Chapters"):
        for s in summaries:
            ts = seconds_to_timestamp(s.get("start") or 0)
            lines = [l.strip().lstrip("-•*").strip() for l in s["summary"].splitlines() if l.strip()]
            title = next((l for l in lines if not l.lower().startswith(skip_prefixes)), lines[0] if lines else "...")
            if len(title) > 100:
                title = title[:97] + "..."
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;gap:1rem;padding:0.5rem 0;"
                f"border-bottom:1px solid rgba(255,255,255,0.05);'>"
                f"<span style='font-family:DM Mono,monospace;font-size:0.75rem;color:#ff4b2b;"
                f"min-width:52px;flex-shrink:0;'>{ts}</span>"
                f"<span style='font-size:0.82rem;color:#c8c4bc;line-height:1.5;'>{html.escape(title)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def reset_quiz():
    st.session_state.quiz_index = 0
    st.session_state.quiz_answers = []
    st.session_state.quiz_answered_current = False
    st.session_state.quiz_selected = None
    st.session_state.quiz_done = False


def render_quiz():
    quiz = st.session_state.quiz
    total = len(quiz)
    idx = st.session_state.quiz_index

    if st.session_state.quiz_done:
        answers = st.session_state.quiz_answers
        correct = sum(1 for i, q in enumerate(quiz) if answers[i] == q["answer"])
        score_pct = int((correct / total) * 100)

        if score_pct == 100:
            emoji, msg = "🏆", "Perfect score!"
        elif score_pct >= 80:
            emoji, msg = "🎉", "Great job!"
        elif score_pct >= 60:
            emoji, msg = "👍", "Not bad!"
        else:
            emoji, msg = "📚", "Keep learning!"

        st.markdown(
            f"<div style='text-align:center;padding:1.5rem 0;'>"
            f"<div style='font-size:2.5rem;margin-bottom:0.5rem;'>{emoji}</div>"
            f"<div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#f0ede6;'>{correct}/{total}</div>"
            f"<div style='font-size:0.8rem;color:#6b6760;margin-top:0.25rem;'>{msg} — {score_pct}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for i, q in enumerate(quiz):
            user_ans = answers[i]
            is_correct = user_ans == q["answer"]
            border_color = "#22c55e" if is_correct else "#ff4b2b"
            icon = "✓" if is_correct else "✗"
            wrong_line = f'<div style="font-size:0.75rem;color:#6b6760;margin-top:0.2rem;">Your answer: {html.escape(user_ans or "No answer")}</div>' if not is_correct else ""
            st.markdown(
                f"<div style='border:1px solid {border_color};border-radius:8px;"
                f"padding:0.8rem 1rem;margin-bottom:0.6rem;'>"
                f"<div style='font-size:0.82rem;color:#f0ede6;margin-bottom:0.3rem;font-weight:600;'>"
                f"Q{i+1}. {html.escape(q['question'])}</div>"
                f"<div style='font-size:0.78rem;color:{border_color};'>{icon} {html.escape(q['answer'])}</div>"
                f"{wrong_line}</div>",
                unsafe_allow_html=True,
            )

        if st.button("Retry Quiz →", key="retry_quiz"):
            reset_quiz()
            st.rerun()
        return

    progress = idx / total
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;'>"
        f"<span style='font-family:DM Mono,monospace;font-size:0.65rem;letter-spacing:0.15em;"
        f"text-transform:uppercase;color:#ff4b2b;'>Question {idx + 1} of {total}</span>"
        f"<span style='font-size:0.7rem;color:#6b6760;'>{int(progress * 100)}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(progress)
    st.markdown("<div style='margin-bottom:1.25rem;'></div>", unsafe_allow_html=True)

    q = quiz[idx]
    answered = st.session_state.quiz_answered_current
    user_ans = st.session_state.quiz_answers[idx] if answered else None

    st.markdown(
        f"<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;"
        f"color:#f0ede6;line-height:1.4;margin-bottom:1.25rem;'>"
        f"{html.escape(q['question'])}</div>",
        unsafe_allow_html=True,
    )

    if not answered:
        selected = st.radio(
            label="options",
            options=q["options"],
            index=None,
            key=f"quiz_radio_{idx}",
            label_visibility="collapsed",
        )
        st.session_state.quiz_selected = selected

        if st.button("Confirm Answer →", key=f"confirm_{idx}", disabled=selected is None):
            st.session_state.quiz_answers.append(selected)
            st.session_state.quiz_answered_current = True
            st.rerun()
    else:
        for opt in q["options"]:
            is_correct_opt = opt == q["answer"]
            is_user = opt == user_ans
            if is_correct_opt:
                bg, border, color = "rgba(34,197,94,0.1)", "#22c55e", "#22c55e"
                suffix = " ✓"
            elif is_user and not is_correct_opt:
                bg, border, color = "rgba(255,75,43,0.1)", "#ff4b2b", "#ff4b2b"
                suffix = " ✗"
            else:
                bg, border, color = "rgba(255,255,255,0.02)", "rgba(255,255,255,0.07)", "#6b6760"
                suffix = ""
            st.markdown(
                f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
                f"padding:0.65rem 1rem;margin-bottom:0.5rem;font-size:0.85rem;color:{color};'>"
                f"{html.escape(opt)}{suffix}</div>",
                unsafe_allow_html=True,
            )

        is_last = idx == total - 1
        btn_label = "See Results →" if is_last else "Next Question →"
        if st.button(btn_label, key=f"next_{idx}"):
            if is_last:
                st.session_state.quiz_done = True
            else:
                st.session_state.quiz_index += 1
                st.session_state.quiz_answered_current = False
                st.session_state.quiz_selected = None
            st.rerun()


def build_download_text(sections: dict) -> str:
    lines = []
    if sections.get("tldr"):
        lines += ["TL;DR", "-" * 40, sections["tldr"].strip(), ""]
    if sections.get("key_points"):
        points = "\n".join(l.strip() for l in sections["key_points"].splitlines() if l.strip())
        lines += ["KEY POINTS", "-" * 40, points, ""]
    if sections.get("action_items"):
        actions = "\n".join(l.strip() for l in sections["action_items"].splitlines() if l.strip())
        lines += ["ACTION ITEMS", "-" * 40, actions, ""]
    return "\n".join(lines)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2.5rem 0 2rem;'>
    <div style='font-family:DM Mono,monospace;font-size:0.7rem;font-weight:500;letter-spacing:0.2em;text-transform:uppercase;color:#ff4b2b;margin-bottom:0.75rem;'>AI-Powered</div>
    <h1 style='font-family:DM Mono,monospace;font-size:clamp(2.4rem,6vw,3.6rem);font-weight:800;line-height:1.05;letter-spacing:-0.02em;color:#f0ede6;margin:0 0 1rem;'>YouTube<br><span style='color:#ff4b2b;'>Summarizer</span></h1>
    <p style='font-size:0.82rem;color:#6b6760;line-height:1.6;max-width:420px;margin:0 auto;text-align:center;'>Paste any YouTube URL and get a structured summary, key points, and action items in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
user_id = get_ip()
remaining = DAILY_SUMMARIZE_LIMIT - summarize_count(user_id)

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if remaining <= 0:
    st.markdown(
        "<div style='background:rgba(255,75,43,0.08);border:1px solid rgba(255,75,43,0.3);"
        "border-radius:8px;padding:0.75rem 1rem;font-size:0.82rem;color:#ff4b2b;margin-top:0.5rem;'>"
        "You've reached your limit of 5 summaries today. Come back tomorrow.</div>",
        unsafe_allow_html=True,
    )
else:
    if remaining <= 2:
        st.markdown(
            f"<div style='font-size:0.72rem;color:#6b6760;margin-top:0.4rem;'>"
            f"{remaining} summarization{'s' if remaining != 1 else ''} remaining today.</div>",
            unsafe_allow_html=True,
        )
    summarize_btn = st.button("Summarize →")

    if summarize_btn:
        if not url.strip():
            st.error("Please enter a YouTube URL.")
        else:
            st.session_state.quiz = None
            st.session_state.qa_answer = None
            st.session_state.qa_count = 0
            reset_quiz()

            with st.spinner("Fetching transcript and summarizing..."):
                try:
                    transcript = fetch_transcript(url.strip())
                    docs, entries = transcript_to_documents(transcript)
                    chunks = chunk_docs(docs, entries)
                    summaries = summarize_chunks(chunks)
                    raw_summary = combine_summaries(summaries)
                    sections = parse_summary(raw_summary)
                    st.session_state.sections = sections
                    st.session_state.summaries = summaries
                    increment_usage(user_id)
                except Exception as e:
                    err = str(e)
                    if "no transcript" in err.lower() or "transcriptsdisabled" in err.lower():
                        st.error("This video doesn't have a transcript available.")
                    elif "invalid" in err.lower() or not any(x in url for x in ["youtube.com", "youtu.be"]):
                        st.error("Couldn't parse that URL. Make sure it's a valid YouTube link.")
                    else:
                        st.error(f"Something went wrong: {err}")

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.sections:
    sections = st.session_state.sections
    summaries = st.session_state.summaries

    render_section("TL;DR", sections["tldr"])
    render_section("Key Points", sections["key_points"])
    render_section("Action Items", sections["action_items"])
    render_chapters(summaries)

    st.download_button(
        label="Download Summary →",
        data=build_download_text(sections),
        file_name="summary.txt",
        mime="text/plain",
    )

    st.markdown("<div style='margin-top:2.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top:1px solid rgba(255,255,255,0.07);margin-bottom:2rem;'></div>",
        unsafe_allow_html=True,
    )

    # ── Q&A ───────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:0.65rem;font-weight:500;"
        "letter-spacing:0.2em;text-transform:uppercase;color:#ff4b2b;margin-bottom:0.75rem;'>Ask the Video</div>",
        unsafe_allow_html=True,
    )

    qa_remaining = QA_LIMIT - st.session_state.qa_count
    if qa_remaining <= 0:
        st.markdown(
            "<div style='background:rgba(255,75,43,0.08);border:1px solid rgba(255,75,43,0.3);"
            "border-radius:8px;padding:0.75rem 1rem;font-size:0.82rem;color:#ff4b2b;'>"
            "You've used all 3 questions for this video.</div>",
            unsafe_allow_html=True,
        )
    else:
        question = st.text_input("Ask anything about this video (Up to 3 Questions)", placeholder="Right here ...", key="qa_input")
        if st.button("Ask →", key="ask_btn"):
            if question.strip():
                with st.spinner("Thinking..."):
                    st.session_state.qa_answer = answer_question(question.strip(), summaries)
                    st.session_state.qa_count += 1

    if st.session_state.qa_answer:
        card("Answer", f"<p style='margin:0;'>{html.escape(st.session_state.qa_answer)}</p>")

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border-top:1px solid rgba(255,255,255,0.07);margin-bottom:2rem;'></div>",
        unsafe_allow_html=True,
    )

    # ── Quiz ──────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:0.65rem;font-weight:500;"
        "letter-spacing:0.2em;text-transform:uppercase;color:#ff4b2b;margin-bottom:0.75rem;'>Quiz</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.quiz is None:
        if st.button("Generate Quiz →", key="gen_quiz"):
            with st.spinner("Generating quiz..."):
                try:
                    st.session_state.quiz = generate_quiz(summaries, num_questions=5)
                    reset_quiz()
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't generate quiz: {e}")
    else:
        render_quiz()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">BUILT WITH LANGCHAIN + OPENAI</div>', unsafe_allow_html=True)