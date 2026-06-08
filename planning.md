# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

My system helps Rice University students both incoming and current find trustworthy insights about off-campus housing. This information is hard to find because apartment complexes market themselves positively, while honest resident experiences are spread across reviews, forums, and social media. It is valuable because students can quickly compare housing options using real feedback from people who have actually lived there.

---

## Documents

| # | Source | Description | File |
|---|--------|-------------|------|
| 1 | Yelp | The Circle at Hermann Park reviews | housing_yelp_01_circle_reviews.txt |
| 2 | Yelp | District at Greenbriar reviews | housing_yelp_02_district_at_Greenbriar_reviews.txt..txt |
| 3 | Yelp | Residences at Gramercy reviews | housing_yelp_03_residences_at_Gramercy_reviews.txt |
| 4 | Yelp | The Maroneal reviews | housing_yelp_04_The_Maroneal_reviews.txt |
| 5 | Yelp | Latitude Med Center reviews | housing_yelp_05_Latitude_reviews.txt |
| 6 | Reddit | Off-campus housing questions (r/riceuniversity) | housing_reddit_01.txt |
| 7 | Reddit | Regrets about off-campus housing (r/riceuniversity) | housing_reddit_02.txt |
| 8 | Reddit | Best places to commute to Rice (r/riceuniversity) | housing_reddit_03.txt |
| 9 | Reddit | Off-campus housing options at Rice (r/riceuniversity) | housing_reddit_04.txt |
| 10 | Reddit | Off-campus housing undergrad (r/riceuniversity) | housing_reddit_05.txt |

---

## Chunking Strategy

**Chunk size:** 3000 characters

**Overlap:** None (reviews are self-contained)

**Reasoning:** I chose 3000 characters to keep each Yelp review intact as a complete unit. Reviews range from 500–3000 characters, so this size captures the full resident opinion without fragmenting it. Since each review is independent and expresses a coherent experience, overlap between chunks is unnecessary. Management responses are excluded to focus purely on resident experiences.

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 (sentence-transformers, local, no API key required)

**Top-k:** 6 chunks per query

**Production tradeoff reflection:** If cost weren't a constraint, I'd prioritize accuracy on short, opinion-based text (reviews are often informal and sarcastic, which can confuse some embeddings) and latency (users expect fast responses). I'd consider larger models like all-mpnet-base-v2 or commercial APIs like OpenAI's embeddings, which may handle domain-specific language better but trade off cost and speed.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What safety issues do students report at The Circle? | Frequent tire theft and break-ins in the parking garage, broken security doors left unlocked, robbery attempts on residents, and poor management response to security concerns. |
| 2 | How quickly does Latitude management respond to resident requests? | Responses are inconsistent and often slow; email responses take 4+ weeks, maintenance issues take weeks to months to resolve (AC problems took 2+ weeks, water issues lasted 3+ months), and lease termination questions require multiple follow-ups over a week to clarify. |
| 3 | What maintenance problems do residents complain about at The Maroneal? | AC failures (taking 2+ weeks to fix, stuck at 79 degrees), frequent water shutdowns (6+ times in 1.5 years), mold in bathrooms, blocked kitchen sinks, water leaks from AC units, broken doorbells and gates, and consistently slow maintenance response times. |
| 4 | What parking issues do students experience at District at Greenbriar? | Tight, difficult-to-navigate garage with frequent theft (catalytic converters, tires, break-ins), limited parking spots, EV charging stations blocked by regular cars, and safety hazards from residents speeding through narrow lanes. |
| 5 | What hidden fees or lease termination charges do students report at Residences at Gramercy? | Excessive move-out charges for normal wear and tear (carpet replacement, cleaning fees $200+), high lease termination fees, unexpected rent charges for additional months after move-out, and unclear or disputed fee explanations from management. |

---

## Anticipated Challenges

1. **Off-topic retrieval:** When a student asks "What do students say about safety at The Circle?", the system might return chunks about parking or management instead if those reviews mention The Circle but focus on different topics. Semantic similarity alone may not distinguish between different subtopics within the same apartment's reviews.

2. **Chunks that split key information across boundaries:** A review might say "The AC was broken for 2 weeks" but if that sentence spans the 3000-character boundary, the system retrieves only "The AC was broken" without the critical duration context, leading to incomplete answers.

3. **Noisy or inconsistent documents:** Reviews contradict each other (one says District at Greenbriar is amazing, another says it's terrible) or contain irrelevant personal rants. The system must retrieve both perspectives, which could confuse users about ground truth or lead to answers that seem uncertain.

---

## Architecture

```
Documents (Yelp, Reddit) 
    ↓
[Ingestion: load files, clean text, exclude management responses]
    ↓
[Chunking: split by review, 3000 characters per chunk]
    ↓
[Embedding: all-MiniLM-L6-v2 → ChromaDB]
    ↓
[Retrieval: semantic search, top-6 similar chunks]
    ↓
[Generation: Groq LLM + grounded system prompt]
    ↓
[Output: answer + source citations]
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
I'll give Claude my Chunking Strategy section (3000 chars, whole reviews, exclude management responses) and a sample Yelp review, then ask it to implement `load_documents()` and `chunk_text()` functions that load all 10 files, clean them, and split by review. I'll verify by printing 5 chunks and checking that each is a complete review with no HTML artifacts or management responses.

**Milestone 4 — Embedding and retrieval:**
I'll provide Claude with the Architecture diagram and Retrieval Approach section (all-MiniLM-L6-v2, top-k=6, ChromaDB), then ask it to implement `embed_and_store()` and `retrieve()` functions. I'll verify by running 3 test queries from my evaluation plan and checking that retrieved chunks are semantically relevant to the question.

**Milestone 5 — Generation and interface:**
I'll give Claude my Evaluation Plan (with expected answers) and ask it to implement a grounded generation function using Groq's LLM with a system prompt that enforces answering only from retrieved context. I'll also ask it to build a Gradio interface with question input and answer + sources output. I'll verify by testing all 5 evaluation questions and confirming responses are grounded with source citations.
