import streamlit as st
import os
from dotenv import load_dotenv
from summarize import summarize_youtube_video
from chunking import fetch_transcript, transcript_to_documents, chunk_docs
from summarize import summarize_chunks, combine_summaries

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Summarizer",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a;
    color: #f0ede6;
    font-family: 'DM Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(255,75,43,0.12) 0%, transparent 70%),
        #0a0a0a;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { display: none; }

.block-container {
    max-width: 720px;
    padding: 3rem 2rem 4rem;
    margin: 0 auto;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 2rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff4b2b;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: #f0ede6;
    margin: 0 0 1rem;
}
.hero-title span {
    color: #ff4b2b;
}
.hero-sub {
    font-size: 0.82rem;
    color: #6b6760;
    line-height: 1.6;
    max-width: 420px;
    margin: 0 auto;
}

/* ── Input card ── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

/* Streamlit input overrides */
[data-testid="stTextInput"] input {
    background: #111 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #f0ede6 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #ff4b2b !important;
    box-shadow: 0 0 0 2px rgba(255,75,43,0.15) !important;
}
[data-testid="stTextInput"] label {
    color: #6b6760 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}

/* Button */
[data-testid="stButton"] button {
    width: 100%;
    background: #ff4b2b !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    margin-top: 0.75rem !important;
}
[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] button:active {
    transform: translateY(0) !important;
}

/* ── Result sections ── */
.result-wrapper {
    margin-top: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
.result-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
}
.result-section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff4b2b;
    margin-bottom: 0.75rem;
}
.result-section-body {
    font-size: 0.88rem;
    line-height: 1.75;
    color: #c8c4bc;
}
.result-section-body ul {
    margin: 0;
    padding-left: 1.2rem;
}
.result-section-body li {
    margin-bottom: 0.4rem;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] {
    color: #ff4b2b !important;
}

/* ── Error ── */
[data-testid="stAlert"] {
    background: rgba(255,75,43,0.08) !important;
    border: 1px solid rgba(255,75,43,0.3) !important;
    border-radius: 8px !important;
    color: #ff4b2b !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.7rem;
    color: #333;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered</div>
    <h1 class="hero-title">YouTube<br><span>Summarizer</span></h1>
    <p class="hero-sub">Paste any YouTube URL and get a structured summary, key points, and action items in seconds.</p>
</div>
""", unsafe_allow_html=True)


# ── Input ──────────────────────────────────────────────────────────────────────
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
summarize = st.button("Summarize →")


# ── Logic ──────────────────────────────────────────────────────────────────────
def parse_summary(raw: str) -> dict:
    """Split the raw summary string into tldr, key_points, action_items."""
    sections = {"tldr": "", "key_points": "", "action_items": ""}
    current = None
    buffer = []

    for line in raw.splitlines():
        l = line.strip()
        if l.lower().startswith("tl;dr") or l.lower().startswith("tldr"):
            if current and buffer:
                sections[current] = "\n".join(buffer).strip()
            current = "tldr"
            # grab inline text after the colon if present
            after = l.split(":", 1)[-1].strip() if ":" in l else ""
            buffer = [after] if after else []
        elif l.lower().startswith("key point"):
            if current and buffer:
                sections[current] = "\n".join(buffer).strip()
            current = "key_points"
            buffer = []
        elif l.lower().startswith("action item"):
            if current and buffer:
                sections[current] = "\n".join(buffer).strip()
            current = "action_items"
            buffer = []
        else:
            if current:
                buffer.append(line)

    if current and buffer:
        sections[current] = "\n".join(buffer).strip()

    return sections


def render_section(label: str, content: str):
    if not content.strip():
        return
    # convert markdown-ish bullets to html list
    lines = content.strip().splitlines()
    html_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip().lstrip("-•*").strip()
        if line.strip().startswith(("-", "•", "*")):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{stripped}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if stripped:
                html_lines.append(f"<p style='margin:0 0 0.3rem'>{stripped}</p>")
    if in_list:
        html_lines.append("</ul>")

    body = "\n".join(html_lines)
    st.markdown(f"""
    <div class="result-section">
        <div class="result-section-label">{label}</div>
        <div class="result-section-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


if summarize:
    if not url.strip():
        st.error("Please enter a YouTube URL.")
    else:
        with st.spinner("Fetching transcript and summarizing..."):
            try:
                transcript = fetch_transcript(url.strip())
                st.write(f"✅ Fetched {len(transcript)} transcript entries")

                docs = transcript_to_documents(transcript)
                st.write(f"✅ Created {len(docs)} documents")

                chunks = chunk_docs(docs)
                st.write(f"✅ Created {len(chunks)} chunks")

                summaries = summarize_chunks(chunks)
                st.write(f"✅ Summarized {len(summaries)} chunks")
                # st.write(summaries[0])
                # st.write(summaries[1])

                raw_summary = combine_summaries(summaries)
                st.write("✅ Combined summaries")

                st.write(raw_summary)
        # with st.spinner("Fetching transcript and summarizing..."):
        #     try:
        #         raw_summary = summarize_youtube_video(url.strip())
        #         sections = parse_summary(raw_summary)
        #
        #         st.markdown('<div class="result-wrapper">', unsafe_allow_html=True)
        #         render_section("TL;DR", sections["tldr"])
        #         render_section("Key Points", sections["key_points"])
        #         render_section("Action Items", sections["action_items"])
        #         st.markdown('</div>', unsafe_allow_html=True)
        #
            except Exception as e:
                err = str(e)
                if "no transcript" in err.lower() or "transcriptsdisabled" in err.lower():
                    st.error("This video doesn't have a transcript available.")
                elif "invalid" in err.lower() or not any(x in url for x in ["youtube.com", "youtu.be"]):
                    st.error("Couldn't parse that URL. Make sure it's a valid YouTube link.")
                else:
                    st.error(f"Something went wrong: {err}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">BUILT WITH LANGCHAIN + OPENAI</div>', unsafe_allow_html=True)