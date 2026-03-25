from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def transcript_to_documents(transcript: list) -> List[Document]:
    docs = []
    for entry in transcript:
        docs.append(
            Document(
                page_content=entry["text"],
                metadata={
                    "source": "youtube",
                    "start": entry.get("start"),
                    "duration": entry.get("duration"),
                }
            )
        )
    return docs

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