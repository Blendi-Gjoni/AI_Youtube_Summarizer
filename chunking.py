from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

def fetch_transcript(url: str) -> List[dict]:
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        return []

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)

    return transcript

def transcript_to_documents(transcript) -> tuple[List[Document], list[dict]]:
    entries = []
    char_pos = 0
    for entry in transcript:
        entries.append({
            "text": entry.text,
            "start": entry.start,
            "char_start": char_pos,
        })
        char_pos += len(entry.text) + 1

    full_text = " ".join([e["text"] for e in entries])
    doc = Document(
        page_content=full_text,
        metadata={"source": "youtube", "start": entries[0]["start"]}
    )
    return [doc], entries

def chunk_docs(documents: List[Document], entries: list[dict]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        start_index = chunk.metadata.get("start_index", 0)

        real_start = entries[0]["start"]
        for entry in entries:
            if entry["char_start"] <= start_index:
                real_start = entry["start"]
            else:
                break

        chunk.metadata["start"] = real_start
        chunk.metadata["chunk_id"] = f"youtube_{start_index}"

    return chunks