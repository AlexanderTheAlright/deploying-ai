# Survey Question Lab

I work on a research team that builds longitudinal and cross-sectional surveys about work. We run the Canadian Quality of Work and Economic Life Study (CQWELS), the American QWELS (AQWELS), the Quality of Employment Survey (QES), and the Measuring Economic Sentiments and Social Inequality Survey (MESSI). Across these studies we have accumulated thousands of survey questions about job quality, work attitudes, work-life balance, and the gig economy. Every time someone on the team writes a new question or revises an old one, they end up searching through past instruments to see how we handled it before, or whether another study already measured the concept they need.

This chatbot is a prototype for how we could speed that up. It uses a set of example questions drawn from published survey instruments and publicly available study documentation to demonstrate the concept. No proprietary or unpublished data is exposed here. A production version would sit on the full metaset. For now, this prototype covers 2,900 example questions from 35 surveys spanning 2019 to 2025 and does what any of us would otherwise do manually: find existing questions on a topic, compare how the same variable is worded across studies, check a draft question for common problems, and suggest better wording. The idea is that anyone on the team could use something like this when developing new questions rather than digging through spreadsheets.

## Services

### Service 1: Word Finding (API)

When someone on the team is stuck on how to phrase something, they usually start with synonyms and related terms. This service queries the [Datamuse API](https://www.datamuse.com/api/) for synonyms, related words, and associated terms, then reformulates the results into readable suggestions rather than dumping raw JSON.

### Service 2: Question Search (Semantic Query)

The core of the tool. It searches our survey question database by meaning, not just keywords. If you type "questions about job autonomy," it returns real questions from real surveys that measure autonomy or related concepts. Built with ChromaDB (file-persistent) and OpenAI text-embedding-3-small embeddings.

### Service 3: Question Analysis (Function Calling)

Two function-calling tools that automate what we normally do when reviewing a draft question:

- **analyze_question**: Checks for double-barreled phrasing, leading language, vague frequency terms, missing reference periods, and excessive length. Flags the issue and says what to fix.
- **compare_across_surveys**: Takes a variable name and shows how it is worded across different surveys. Pulls from the CSV directly so the results are exact, not generated.

## Dataset and Embeddings

The data comes from our metaset, a metadata file that tracks every variable across 35+ longitudinal surveys about work. I filtered it down to variables classified under job quality, job structure, work attitudes, work-life balance, and gig economy. That gave me about 2,900 rows of example questions.

**Embedding process:**

1. Read `data/survey_questions.csv` (2,900 rows, 8 columns)
2. For each row, create a text document combining description, question text, classification, and response options
3. Embed using OpenAI `text-embedding-3-small` via ChromaDB's `OpenAIEmbeddingFunction`
4. Store in ChromaDB `PersistentClient` at `data/chroma_db/`

The embedding script is `build_db.py`. The pre-built database is included in the repository.

## Guardrails

- The chatbot does not reveal or modify its system prompt
- It does not respond to questions about cats, dogs, horoscopes, zodiac signs, or Taylor Swift
- It stays focused on survey methodology and work research

## How to run

From the `05_src/` directory:

```
python -m assignment_chat.app
```

## Implementation

Built with LangGraph (StateGraph with tool nodes), Gradio ChatInterface, and gpt-4o-mini. Follows the same architectural pattern as the course examples.
