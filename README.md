# The Unofficial Guide: Off-Campus Housing at Rice

A RAG system that makes student-generated housing knowledge searchable and answerable. Students ask plain-language questions about Rice off-campus apartments and get grounded answers from real resident reviews.

## Demo Video

See the system in action: **[demo.mp4](demo.mp4)**

The video shows:
- 3+ different queries with source citations
- Strong retrieval example (District at Greenbriar parking)
- Weak retrieval example (Circle safety) with failure analysis
- Correct refusal to hallucinate (Gramercy fees not in corpus)
- Walkthrough of evaluation results and failure case analysis

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
cp .env.example .env
# Add your Groq API key to .env

# Run the web interface
python3 src/app.py
# Opens at http://localhost:7860
```

Ask questions like:
- "What do students say about safety at The Circle?"
- "How responsive is Latitude management?"
- "What maintenance issues do residents complain about at The Maroneal?"

## Domain

This system helps Rice University students (incoming and current) find trustworthy insights about off-campus apartments. This knowledge is valuable but hard to find because apartment complexes market themselves positively on official websites, while honest resident experiences are scattered across Yelp reviews, Reddit threads, and housing forums. By aggregating real voices into one searchable system, students can quickly compare apartments based on evidence from people who have actually lived there.

## Document Sources

| # | Source | Type | Details |
|---|--------|------|---------|
| 1 | Yelp | Reviews | The Circle at Hermann Park (28 reviews) |
| 2 | Yelp | Reviews | District at Greenbriar (40+ reviews) |
| 3 | Yelp | Reviews | Residences at Gramercy (20+ reviews) |
| 4 | Yelp | Reviews | The Maroneal (15+ reviews) |
| 5 | Yelp | Reviews | Latitude Med Center (12+ reviews) |
| 6 | Reddit | Discussions | Off-campus housing Q&A, r/riceuniversity |
| 7 | Reddit | Discussions | People who lived off-campus: regrets/experiences |
| 8 | Reddit | Discussions | Best places to live near Rice |
| 9 | Reddit | Discussions | Off-campus housing options at Rice |
| 10 | Reddit | Discussions | Off-campus housing for undergrads |

## Chunking Strategy

**Chunk size:** 3000 characters (whole reviews kept intact when under this limit)

**Overlap:** None (reviews are self-contained; overlap not needed)

**Why these choices fit your documents:**
- Yelp reviews are naturally self-contained opinion units (500–3000 chars each)
- Keeping whole reviews preserves nuance: a single review describes multiple issues (safety, management, maintenance) that together form the resident's full perspective
- Reddit discussions are single comments/threads; splitting them would lose conversational context
- No overlap needed since each review is independent — readers don't need fragment overlap to understand the next review

**Preprocessing:**
- Removed "Business owner information" sections and management responses (boilerplate, no resident signal)
- Removed "Upvote/Downvote" metrics and navigation elements
- Stripped HTML entities (&amp;, &nbsp;, &#39;, etc.)
- Preserved resident names, dates, ratings, and location for context

**Final chunk count:** 155 chunks across 10 documents

## Embedding Model

**Model used:** all-MiniLM-L6-v2 (sentence-transformers, local)
- Runs on CPU, no API key required
- 384-dimensional embeddings
- Fast inference (< 1 second for 155 chunks)

**Production tradeoff reflection:**
If cost were not a constraint, I'd prioritize accuracy on domain-specific language (housing reviews contain domain jargon like "move-out fees," "lease termination," "maintenance request response time" that smaller models may conflate). Larger models like all-mpnet-base-v2 or commercial APIs like OpenAI embeddings would better distinguish between "responsive to online reviews" and "responsive to maintenance requests" — a nuance that tripped up Q2 in evaluation. The tradeoff: accuracy vs. latency + cost. For a real system serving hundreds of students, I'd invest in a larger model and accept the API cost.

## Grounded Generation

**System prompt grounding instruction:**
```
You must answer using ONLY the information provided in the retrieved documents below. 
Do not use any information from your training data or general knowledge.

If the documents don't contain enough information to answer the question, say: 
"I don't have enough information about that based on the available reviews."

Always cite which document(s) your answer comes from.
```

**How source attribution is surfaced:**
- Model is instructed to cite sources inline: `(Source: housing_yelp_01_circle_reviews.txt)`
- Retrieved chunk file names are appended after generation as a "Sources:" section
- Example: "...according to residents (Source: housing_yelp_02_district_at_Greenbriar_reviews.txt..txt)"

## Evaluation Report

| # | Question | Expected answer | System response | Retrieval quality | Response accuracy |
|---|----------|-----------------|-----------------|-------------------|-------------------|
| 1 | What safety issues do students report at The Circle? | Tire theft, break-ins, robbery, poor management response to security concerns | Mentions biking safety through Hermann Park; misses Circle-specific issues | Partially relevant (got safety topic, wrong apartment) | Inaccurate (weak) |
| 2 | How quickly does Latitude management respond to resident requests? | Slow: 4+ weeks for email, months for maintenance (AC 2+ weeks, water 3+ months) | Says responsive; cites manager availability and fast review responses | Relevant (correct apartment) | Partially accurate (contradicts expectation) |
| 3 | What maintenance problems do residents complain about at The Maroneal? | AC failures (2+ weeks), water shutdowns (6+ times), mold, blocked sinks, leaks, slow repairs | Lists old carpet, mold, elevators, pipes, months for repairs | Relevant | Accurate |
| 4 | What parking issues do students experience at District at Greenbriar? | Tight garage, theft, limited spots, EV stations blocked, speeding cars | Tight garage, theft, EV issues, lack of management enforcement | Relevant | Accurate |
| 5 | What hidden fees or lease termination charges do students report at Residences at Gramercy? | High termination fees, move-out charges, $200+ cleaning, normal wear-and-tear disputes | Correctly refuses: "I don't have enough information" | N/A (correct refusal) | Accurate |

**Retrieval quality:** Measured by whether top chunks were from the correct apartment and topic  
**Response accuracy:** Measured against expected answer from evaluation plan

## Failure Case Analysis

**Question that failed:** Q1: "What safety issues do students report at The Circle?"

**What the system returned:** "According to the available reviews, students report concerns about biking through Hermann Park when it's dark... However, there is no specific incident or issue reported at The Circle itself."

**Root cause (tied to pipeline stage):** RETRIEVAL failure. The embedding model weighted "safety" (the query topic) more heavily than "The Circle" (the apartment name). Top retrieved chunk (distance 0.477) was a Reddit discussion about general Hermann Park safety for women at night, not Circle-specific security issues. Circle security reviews (tire theft, broken doors, robbery attempts) had higher distance scores (0.5+), so they ranked below the Hermann Park discussion.

**Why this happened:** 
- Semantic embeddings are topic-first, not proper-noun-first
- The chunk "I can't say specifically, but generally anywhere off campus is not going to be very safe for women after dark" matched the query intent ("safety concerns") more closely than isolated sentences about tire theft
- Apartment names are low-signal for embeddings: they're single noun mentions, not contextual phrases

**What I would change to fix it:** 
- Hybrid search: Combine semantic similarity with BM25 keyword matching on apartment names to boost exact-name matches
- Metadata filtering: Allow users to filter by apartment name, which would retrieve only Circle reviews before re-ranking by relevance
- Longer context: Include apartment name in the chunk metadata and weight it during retrieval

## Stretch Feature: Hybrid Search (Implemented)

**What it does:** Combines semantic search (embeddings) with BM25 keyword matching using reciprocal rank fusion (RRF). For each query, it runs both retrieval methods separately and merges results by combined score: `hybrid_score = 1/(semantic_rank+1) + 1/(bm25_rank+1)`.

**How to use it:**
```python
from src.embedding import RAGSystem
rag = RAGSystem()
rag.embed_and_store()
results = rag.retrieve_hybrid("What safety issues do students report at The Circle?")
# Returns same format as retrieve(): list of dicts with text, source, chunk_id
```

**Why it doesn't help for this dataset:**
- Tested hybrid search on all 5 evaluation questions
- Expected improvement: boost apartment-specific Yelp reviews by matching exact apartment names
- Actual result: **No improvement on Q1, regression on Q2-Q4** (hybrid returned fewer correct sources than semantic alone)
- Root cause: BM25 retrieves based on word frequency, not semantic relevance. Generic mentions of "management," "maintenance," etc. in unrelated apartments ranked high in BM25 and polluted the merged results, crowding out true matches
- Lesson: Hybrid search helps when semantic and keyword searches find different relevant results. Here, both searches struggle with the same problem (topic > proper noun relevance), and merging them just adds noise

**Implementation details:**
- Uses `rank_bm25` library (BM25Okapi algorithm) on tokenized chunk text
- Initialized at RAGSystem construction for performance
- Reciprocal rank fusion with equal weights (1.0x each): prevents one method from dominating
- Handles chunks found by only one method (assigns worst-rank if not in top-k)

## Spec Reflection

**One way the spec helped during implementation:**

The chunking strategy section forced me to think through the document structure before coding. I had read the reviews and noticed they were naturally self-contained units (each review has one resident's full opinion). Writing down "3000 characters, no overlap, whole reviews" in planning.md made me commit to preserving that structure rather than naively splitting by fixed character boundaries. When I built ingestion.py, this spec meant I could write a split_into_reviews() function instead of a dumb character splitter. That design choice prevented the failure mode where a key fact (like "AC broken for 2 weeks") gets split across chunk boundaries and only half the information is retrieved.

**One way implementation diverged from the spec, and why:**

The plan said "exclude management responses" but didn't specify how thoroughly. In early iterations, I was removing "Business owner information" sections but missing manager names at chunk boundaries ("Simpson P.", "Community M.", etc.). Implementation required iterative refinement of the regex patterns because the documents were dirtier than planned. This taught me that cleaning is not a one-pass operation: I had to inspect actual chunks and refine the cleaning rules. The spec assumed a simpler document structure than reality provided.

## AI Usage

**Instance 1: Embedding & Retrieval Code**

- *What I gave the AI:* Planning.md (Retrieval Approach section), the architecture diagram, and a note that I was using ChromaDB
- *What it produced:* A complete embedding.py with RAGSystem class, embed_and_store() to load chunks and embed with sentence-transformers, and retrieve() to query ChromaDB and return top-6 chunks
- *What I changed or overrode:* The AI used ChromaDB's client directly. I had to debug the API (client.chat.completions vs. client.messages) because ChromaDB's Python interface is different from what the AI expected. No logic changes, just API adjustments.

**Instance 2: Generation & Grounding**

- *What I gave the AI:* Planning.md (Evaluation Plan with expected answers), the grounding requirement ("answers from retrieved context only"), and the output format (answer + sources)
- *What it produced:* generation.py with a GroundedRAG class that retrieves chunks, formats them as context, passes them to Groq with a system prompt, and returns answer + sources
- *What I changed or overrode:* The AI's first attempt used a separate `system=` parameter in the Groq call. I had to correct it to pass the system prompt as a message (OpenAI API vs. proprietary Groq API). The grounding logic itself worked as intended: the model correctly refused to answer Q5 and cited sources for Q2-Q4.

---

**Built with:** sentence-transformers, ChromaDB, Groq, Gradio, Python  
**Last updated:** June 2026
