from llm import llm
from langchain_core.documents import Document

def summarize_chunks(chunks: list[Document]) -> list[Document]:
    summaries = []
    for chunk in chunks:
        response = llm.invoke(f"""
            You are summarizing a section of a YouTube video transcript.
            Extract the key points from this section concisely.
            Do not mention that this is a transcript or a video section.
            
            Transcript section:
            {chunk.page_content}
        """)
        summaries.append({
            "summary": response.content,
            "start": chunk.metadata.get("start"),
            "chunk_id": chunk.metadata.get("chunk_id"),
        })

    return summaries

def combine_summaries(summaries: list[Document]) -> str:
    joined = "\n\n".join([s["summary"] for s in summaries])
    response = llm.invoke(f"""
        You are combining partial summaries of a YouTube video into one final summary.
        - Remove redundancy
        - Preserve the narrative flow
        - Structure your output as:
            TL;DR: (2-3 sentences)
            Key Points: (bullet list)
            Action Items: (if any, otherwise omit)
        
        Partial summaries:
        {joined}
    """)
    return response.content