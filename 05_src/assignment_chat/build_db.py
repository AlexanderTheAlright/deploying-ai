"""
Builds the ChromaDB vector database from survey_questions.csv.
Run once from 05_src/. Pre-built database is in the repo.
"""

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import os

load_dotenv(".secrets")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "survey_questions.csv")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")


def build_document(row: pd.Series) -> str:
    """Create a text document from a survey question row."""
    parts = []
    if row["description"]:
        parts.append(f"Description: {row['description']}")
    if row["question"]:
        parts.append(f"Question: {row['question']}")
    if row["classification"]:
        parts.append(f"Classification: {row['classification']}")
    if row["responses"]:
        parts.append(f"Response options: {row['responses']}")
    return "\n".join(parts)


def main():
    print(f"Reading CSV from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.fillna("")
    print(f"Loaded {len(df)} rows")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if it exists
    try:
        client.delete_collection("survey_questions")
        print("Deleted existing collection")
    except Exception:
        pass

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    collection = client.create_collection(
        name="survey_questions",
        embedding_function=embedding_fn,
    )

    # Build documents and metadata
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        doc = build_document(row)
        if not doc.strip():
            continue

        documents.append(doc)
        metadatas.append({
            "surveyid": str(row["surveyid"]),
            "varname": str(row["varname"]),
            "classification": str(row["classification"]),
            "variable": str(row["variable"]),
        })
        ids.append(f"{row['surveyid']}_{row['variable']}_{idx}")

    # ChromaDB has a batch size limit, so add in chunks
    batch_size = 500
    total = len(documents)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"Embedding batch {start}-{end} of {total}...")
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"Done. {total} documents embedded and stored at {CHROMA_PATH}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
