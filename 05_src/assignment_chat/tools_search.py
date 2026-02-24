from langchain.tools import tool
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
from dotenv import load_dotenv
from utils.logger import get_logger

_logs = get_logger(__name__)

load_dotenv(".secrets")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(
    name="survey_questions",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    ),
)


@tool
def search_survey_questions(query: str, n_results: int = 5) -> str:
    """
    Searches the survey question database by meaning. Returns real
    questions from longitudinal studies (QWELS, AQWELS, QES, MESSI)
    that match the query topic.

    Args:
        query: What to search for (e.g., "job autonomy", "workplace
            respect", "financial strain").
        n_results: Number of results to return (default 5, max 10).
    """
    n_results = min(n_results, 10)

    _logs.debug(f"Searching survey questions: '{query}' (n={n_results})")

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    if not results["documents"][0]:
        return f"No survey questions found matching '{query}'."

    lines = [f"Survey questions matching '{query}':\n"]

    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        lines.append(f"--- Result {i + 1} ---")
        lines.append(f"Variable: {meta.get('varname', 'N/A')}")
        lines.append(f"Survey: {meta.get('surveyid', 'N/A')}")
        lines.append(f"Classification: {meta.get('classification', 'N/A')}")
        lines.append(f"Content: {doc}")
        lines.append("")

    return "\n".join(lines)
