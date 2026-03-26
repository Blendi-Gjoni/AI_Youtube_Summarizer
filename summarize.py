from llm import llm
from chunking import fetch_transcript, chunk_docs, transcript_to_documents
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed


def summarize_chunks(chunks: list[Document]) -> list[dict]:
    summaries = [None] * len(chunks)

    def summarize_one(i, chunk):
        response = llm.invoke(f"""
            You are summarizing a section of a YouTube video transcript.
            Extract the key points from this section concisely as bullet points.
            Do not mention that this is a transcript or a video section.

            Transcript section:
            {chunk.page_content}
        """)
        return i, {
            "summary": response.content,
            "start": chunk.metadata.get("start"),
            "chunk_id": chunk.metadata.get("chunk_id"),
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(summarize_one, i, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            i, result = future.result()
            summaries[i] = result

    return summaries


def combine_summaries(summaries: list[dict]) -> str:
    joined = "\n\n".join([s["summary"] for s in summaries])
    response = llm.invoke(f"""
        You are combining partial summaries of a YouTube video into one final summary.
        Remove redundancy and preserve narrative flow.

        You MUST respond with only a valid JSON object. No markdown, no backticks, no extra text before or after.
        Use exactly this structure:
        {{
            "tldr": "2-3 sentence summary here",
            "key_points": ["point 1", "point 2", "point 3", ...],
            "action_items": ["action 1", "action 2", ...]
        }}

        If there are no action items, return an empty list for that field.

        Partial summaries:
        {joined}
    """)
    return response.content

def summarize_youtube_video(url: str):
    transcript = fetch_transcript(url)
    docs, entries = transcript_to_documents(transcript)
    chunks = chunk_docs(docs, entries)
    summaries = summarize_chunks(chunks)
    final_summary = combine_summaries(summaries)
    return final_summary, summaries