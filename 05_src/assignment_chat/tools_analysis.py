from langchain.tools import tool
import pandas as pd
import os
import re
from utils.logger import get_logger

_logs = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "survey_questions.csv")


@tool
def analyze_question(question_text: str) -> str:
    """
    Analyzes a draft survey question for common methodological problems.
    Checks for double-barreled questions, leading language, ambiguous terms,
    excessive length, and missing reference periods.

    Args:
        question_text: The survey question text to analyze.
    """
    issues = []

    # double-barreled check
    conjunctions = re.findall(r'\b(and|or)\b', question_text.lower())
    question_marks = question_text.count('?')
    if len(conjunctions) >= 2 or question_marks > 1:
        issues.append(
            "DOUBLE-BARRELED: Looks like this asks about more than one thing. "
            "When you join concepts with 'and' or 'or', respondents have to pick "
            "one to answer about. Split it."
        )

    word_count = len(question_text.split())
    if word_count > 30:
        issues.append(
            f"TOO LONG: {word_count} words. Past 25 or 30, respondents start skimming. "
            "Move background info into a stem or instruction block and keep the "
            "question itself short."
        )

    leading_phrases = [
        r"\bdon't you think\b",
        r"\bisn't it true\b",
        r"\bwouldn't you agree\b",
        r"\bof course\b",
        r"\bobviously\b",
        r"\bclearly\b",
        r"\beveryone knows\b",
        r"\bmost people\b",
        r"\bshould\b.*\b(agree|think|believe)\b",
    ]
    for pattern in leading_phrases:
        if re.search(pattern, question_text.lower()):
            issues.append(
                "LEADING: There is evaluative language here that tips the respondent "
                "toward a particular answer. Strip it out. The response scale does "
                "the work of capturing their position."
            )
            break

    ambiguous = [
        r"\boften\b", r"\busually\b", r"\bsometimes\b", r"\brarely\b",
        r"\brecently\b", r"\ba lot\b", r"\bregularly\b",
    ]
    found_ambiguous = []
    for pattern in ambiguous:
        match = re.search(pattern, question_text.lower())
        if match:
            found_ambiguous.append(match.group())
    if found_ambiguous:
        issues.append(
            f"VAGUE FREQUENCY: '{', '.join(found_ambiguous)}' means something "
            "different to every respondent. Replace with a concrete time frame "
            "('in the past 7 days', 'once a week or more') or put frequency "
            "into the response options."
        )

    time_refs = re.findall(
        r'\b(past|last|current|this|previous|ago|week|month|year|days)\b',
        question_text.lower()
    )
    if not time_refs and word_count > 5:
        issues.append(
            "NO REFERENCE PERIOD: There is no time frame here. One respondent "
            "answers about today, another about the past five years, and you "
            "cannot compare them. Add something like 'in the past month' or "
            "'in your current job'."
        )

    negatives = re.findall(r'\b(not|never|no|none|neither|nor)\b', question_text.lower())
    if len(negatives) >= 2:
        issues.append(
            "DOUBLE NEGATIVE: Two negations and respondents lose track of which "
            "direction they are answering in. Rephrase with positive wording or "
            "at most one negation."
        )

    if not issues:
        return (
            "This looks clean. No double-barreling, no leading language, "
            "reasonable length. You should still pilot it with a few respondents "
            "before fielding."
        )

    header = f"Analysis of: \"{question_text}\"\n\nIssues found ({len(issues)}):\n\n"
    return header + "\n\n".join(f"{i+1}. {issue}" for i, issue in enumerate(issues))


@tool
def compare_across_surveys(varname: str) -> str:
    """
    Shows how the same variable (by varname) is measured across different
    surveys. Displays the question text, response options, and survey source
    for each instance.

    Args:
        varname: The variable name to look up (e.g., 'b25_respect', 'jobsat').
    """
    if not os.path.exists(CSV_PATH):
        return "Survey data file not found."

    df = pd.read_csv(CSV_PATH)
    matches = df[df['varname'].str.lower() == varname.lower()]

    if matches.empty:
        # Try partial match
        partial = df[df['varname'].str.lower().str.contains(varname.lower())]
        if partial.empty:
            return (
                f"No variables found matching '{varname}'. "
                f"Try a different name or search the database by topic instead."
            )
        # Show partial matches as suggestions
        suggestions = partial['varname'].unique()[:10]
        return (
            f"No exact match for '{varname}'. Similar variable names: "
            f"{', '.join(suggestions)}"
        )

    lines = [f"Variable '{varname}' across {len(matches)} survey instances:\n"]

    for _, row in matches.iterrows():
        lines.append(f"Survey: {row['surveyid']}")
        lines.append(f"  Description: {row['description']}")
        if row['question']:
            lines.append(f"  Question: {row['question']}")
        if row['responses']:
            lines.append(f"  Responses: {row['responses']}")
        lines.append("")

    return "\n".join(lines)
