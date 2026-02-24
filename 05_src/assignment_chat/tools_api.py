from langchain.tools import tool
import requests
import json
from utils.logger import get_logger

_logs = get_logger(__name__)


@tool
def find_related_words(word: str, relation_type: str = "related") -> str:
    """
    Finds words related to the input word using the Datamuse API.
    Useful when refining survey question wording.

    Args:
        word: The word to find relations for.
        relation_type: One of "synonyms", "related", or "associated".
            - "synonyms": words with similar meaning
            - "related": words in the same general space
            - "associated": words that tend to appear alongside the input
    """
    param_map = {
        "synonyms": "rel_syn",
        "related": "ml",
        "associated": "rel_trg",
    }
    param = param_map.get(relation_type, "ml")

    url = "https://api.datamuse.com/words"
    params = {param: word, "max": 15}

    _logs.debug(f"Querying Datamuse: {param}={word}")

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as e:
        _logs.error(f"Datamuse API error: {e}")
        return f"Could not reach the word-finding service. Error: {e}"

    if not results:
        return f"No {relation_type} words found for '{word}'."

    words = [item["word"] for item in results]
    output = f"{relation_type.capitalize()} for '{word}': {', '.join(words)}"
    return output
