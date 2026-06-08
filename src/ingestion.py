import os
import re
from pathlib import Path
from typing import List, Dict

def load_documents(documents_dir: str = "documents") -> Dict[str, str]:
    """
    Load all .txt files from the documents directory.
    Returns a dict of {filename: raw_text}
    """
    documents = {}
    docs_path = Path(documents_dir)

    for file_path in sorted(docs_path.glob("*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[file_path.name] = f.read()

    print(f"Loaded {len(documents)} documents")
    return documents


def clean_document(text: str) -> str:
    """
    Clean a single document by removing:
    - Management/business owner responses
    - HTML tags and entities
    - Navigation text, boilerplate headers
    - Extra whitespace
    """
    # Remove all management response sections (Business owner, Business Manager, Manager, etc.)
    # Match from "Business owner" or "Manager" headers through the response text
    text = re.sub(
        r'(?:Business owner|Business Manager|Karen M\.|Jessie W\.|Simpson P\.|Community M\.|Property M\.|Sky U\.) .*?(?=\n[A-Z][a-z]+ [A-Z]\.?\n|\nu/|\n# |$)',
        '',
        text,
        flags=re.DOTALL
    )

    # Also remove "Upvote/Downvote" and voting elements
    text = re.sub(r'Upvote\s+\d+\s+Downvote\s+\d+.*?Share', '', text, flags=re.DOTALL)
    text = re.sub(r'Upvote\s+\d+\s+Downvote.*?(?=\n[A-Z]|$)', '', text, flags=re.DOTALL)

    # Remove common boilerplate
    text = re.sub(r'Read more', '', text)
    text = re.sub(r'See all photos.*?\n', '', text, flags=re.DOTALL)

    # Remove manager/business owner names and titles at the end of chunks
    text = re.sub(r'\n+(?:Simpson P\.|Community M\.|Property M\.|Jessie W\.|Karen M\.|Sky U\.|Sky Usher|Morgan Group|Business Owner|Business Manager|Business Customer Service|Reply|Award|Share)\s*$', '', text, flags=re.MULTILINE)

    # Remove HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&#39;', "'")
    text = text.replace('&quot;', '"')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')

    # Remove HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # max 2 newlines between paragraphs
    text = re.sub(r' {2,}', ' ', text)       # max 1 space between words

    return text.strip()


def chunk_documents(documents: Dict[str, str], chunk_size: int = 3000) -> List[Dict]:
    """
    Split documents into chunks following the spec:
    - 3000 characters per chunk (captures whole reviews)
    - No overlap (reviews are self-contained)
    - Keep reviews intact (don't split mid-review)

    Returns list of dicts: {text, source, chunk_id}
    """
    chunks = []

    for source_filename, raw_text in documents.items():
        # Clean the document
        cleaned_text = clean_document(raw_text)

        # Split by reviewing patterns: look for reviewer names followed by review text
        # Yelp reviews start with name/location and end before the next name or "Business owner"
        reviews = split_into_reviews(cleaned_text)

        chunk_id = 0
        for review_text in reviews:
            review_text = review_text.strip()

            # Skip empty reviews
            if len(review_text) < 50:
                continue

            # If review is smaller than chunk_size, keep it as one chunk
            if len(review_text) <= chunk_size:
                chunks.append({
                    'text': review_text,
                    'source': source_filename,
                    'chunk_id': chunk_id
                })
                chunk_id += 1
            else:
                # If review exceeds chunk_size, split it into multiple chunks
                # This is rare but handles very long reviews
                for i in range(0, len(review_text), chunk_size):
                    chunk_text = review_text[i:i + chunk_size].strip()
                    if len(chunk_text) > 50:
                        chunks.append({
                            'text': chunk_text,
                            'source': source_filename,
                            'chunk_id': chunk_id
                        })
                        chunk_id += 1

    return chunks


def split_into_reviews(text: str) -> List[str]:
    """
    Split cleaned text into individual reviews.
    Yelp reviews are separated by reviewer names (all caps or title case at line start).
    Reddit threads are separated by username patterns.
    """
    # Pattern for Yelp: Name, Location, Rating, Date
    # Example: "Natasha M.\nWA, WA\n11214\nMay 2, 2026"
    yelp_pattern = r'(?=\n[A-Z][a-z]+ [A-Z]\.?\n[A-Za-z\s,]+\n\d+\n[A-Za-z]+ \d+, \d+)'

    # Pattern for Reddit: username with avatar or bullet point
    reddit_pattern = r'(?=\nu/\w+|(?=\n[a-z_]+\n•))'

    # Try splitting by Yelp pattern first, then Reddit pattern
    if 'Yelp' in text or re.search(r'[A-Z][a-z]+ [A-Z]\.\n[A-Za-z\s,]+\n\d+\n[A-Za-z]+', text):
        reviews = re.split(yelp_pattern, text)
    else:
        # For Reddit, split by username or bullet point markers
        reviews = re.split(r'\n(?:u/\w+|•)', text)

    # Filter out empty reviews
    reviews = [r.strip() for r in reviews if r.strip()]

    return reviews


if __name__ == "__main__":
    # Load and chunk documents
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Chunk size range: {min(len(c['text']) for c in chunks)} - {max(len(c['text']) for c in chunks)} characters")

    # Display 5 sample chunks for inspection
    print("\n" + "="*80)
    print("SAMPLE CHUNKS (5 random examples)")
    print("="*80)

    import random
    sample_chunks = random.sample(chunks, min(5, len(chunks)))

    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\n--- CHUNK {i} ---")
        print(f"Source: {chunk['source']}")
        print(f"Length: {len(chunk['text'])} characters")
        print(f"Preview (first 300 chars):")
        print(chunk['text'][:300])
        if len(chunk['text']) > 300:
            print("...")
