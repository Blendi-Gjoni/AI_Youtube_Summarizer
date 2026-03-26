# YouTube Summarizer

An AI-powered tool that takes any YouTube URL and returns a structured summary with a TL;DR, key points, action items, and timestamped chapters.

Built with LangChain, OpenAI, and Streamlit.

Available at: 

---

## How It Works

1. Fetches the transcript from a YouTube video
2. Joins the transcript into a single document and maps character positions back to timestamps
3. Splits the document into chunks using `RecursiveCharacterTextSplitter`
4. Summarizes each chunk in parallel using GPT-4o-mini (map step)
5. Combines all chunk summaries into a final structured summary (reduce step)
6. Displays the result in a Streamlit UI with a downloadable `.txt` export

---

## Project Structure

```
youtube-summarizer/
├── app.py              # Streamlit UI
├── chunking.py         # Transcript fetching, document creation, chunking
├── summarize.py        # LLM map + reduce summarization
├── llm.py              # OpenAI LLM initialization
├── main.py             # CLI entry point
├── requirements.txt
└── .env                # API key (not committed)
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Blendi-Gjoni/AI_Youtube_Summarizer
cd youtube-summarizer
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI API key**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

---

## Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste a YouTube URL and hit **Summarize**.

To run as a CLI instead:
```bash
python main.py
```

---

## Requirements

```
streamlit
langchain-core
langchain-text-splitters
langchain-openai
youtube-transcript-api
python-dotenv
openai
```

---

## Notes

- Videos without transcripts (auto-generated or manual) will return an error
- Very long videos (2h+) may take 30–60 seconds to process
- Chunk size is set to 4000 characters with 400 character overlap — tunable in `chunking.py`
