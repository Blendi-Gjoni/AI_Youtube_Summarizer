import html
import json
import streamlit as st
from dotenv import load_dotenv
from chunking import fetch_transcript, transcript_to_documents, chunk_docs
from summarize import summarize_chunks, combine_summaries

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
.footer{text-align:center;margin-top:3rem;font-size:0.7rem;color:#333;letter-spacing:0.1em;}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "sections" not in st.session_state:
    st.session_state.sections = None
if "summaries" not in st.session_state:
    st.session_state.summaries = None


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
        # strip markdown code fences if present
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
    except Exception as e:
        # fallback: return the raw text in tldr so something always shows
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


def build_download_text(sections: dict) -> str:
    lines = []
    if sections.get("tldr"):
        lines += ["TL;DR", "-" * 40, sections["tldr"].strip(), ""]
    if sections.get("key_points"):
        points = "\n".join(
            l.strip() for l in sections["key_points"].splitlines() if l.strip()
        )
        lines += ["KEY POINTS", "-" * 40, points, ""]
    if sections.get("action_items"):
        actions = "\n".join(
            l.strip() for l in sections["action_items"].splitlines() if l.strip()
        )
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
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
summarize_btn = st.button("Summarize →")

# ── Summarize ──────────────────────────────────────────────────────────────────
if summarize_btn:
    if not url.strip():
        st.error("Please enter a YouTube URL.")
    else:
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

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">BUILT WITH LANGCHAIN + OPENAI</div>', unsafe_allow_html=True)