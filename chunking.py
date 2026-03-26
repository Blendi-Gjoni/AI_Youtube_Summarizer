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

def transcript_to_documents(transcript) -> List[Document]:
    full_text = " ".join([entry.text for entry in transcript])
    first_start = transcript[0].start if transcript else 0

    return [Document(
        page_content=full_text,
        metadata={
            "source": "youtube",
            "start": first_start,
        }
    )]

def chunk_docs(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=400,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', '')}_{chunk.metadata.get('start_index', 0)}"
    return chunks