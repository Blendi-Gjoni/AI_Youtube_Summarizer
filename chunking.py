from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def transcript_to_documents(transcript: list) -> List[Document]:
    full_text = " ".join([entry["text"] for entry in transcript])
    first_start = transcript[0].get("start", 0) if transcript else 0

    return [Document(
        page_content=full_text,
        metadata={
            "source": "youtube",
            "start": first_start,
        }
    )]

def chunk_docs(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', '')}_{chunk.metadata.get('start_index', 0)}"
    return chunks