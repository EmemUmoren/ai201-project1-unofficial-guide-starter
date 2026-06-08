"""
Inspect chunks for quality: are they clean, complete, and standalone?
"""
from ingestion import load_documents, chunk_documents
import random

documents = load_documents()
chunks = chunk_documents(documents)

print(f"Total chunks: {len(chunks)}")
print(f"Documents loaded: {len(documents)}")
print(f"\nChunk size distribution:")
print(f"  Min: {min(len(c['text']) for c in chunks)}")
print(f"  Max: {max(len(c['text']) for c in chunks)}")
print(f"  Median: {sorted([len(c['text']) for c in chunks])[len(chunks)//2]}")

# Check for common issues
print("\n" + "="*80)
print("QUALITY CHECKS")
print("="*80)

# Check for HTML artifacts
html_chunks = [c for c in chunks if '<' in c['text'] or '&' in c['text']]
print(f"\n✓ Chunks with HTML artifacts: {len(html_chunks)} (should be 0)")

# Check for management response keywords (should be excluded)
mgmt_chunks = [c for c in chunks if 'Business owner' in c['text'] or 'Manager' in c['text'] and 'Manager:' in c['text']]
print(f"✓ Chunks with management responses: {len(mgmt_chunks)} (should be 0)")

# Check for empty or tiny chunks
tiny_chunks = [c for c in chunks if len(c['text']) < 100]
print(f"✓ Chunks under 100 chars: {len(tiny_chunks)}")

# Display detailed samples
print("\n" + "="*80)
print("DETAILED INSPECTION: 5 RANDOM CHUNKS")
print("="*80)

sample_indices = random.sample(range(len(chunks)), min(5, len(chunks)))

for i, idx in enumerate(sample_indices, 1):
    chunk = chunks[idx]
    print(f"\n{'='*80}")
    print(f"CHUNK {idx} (Index {idx} of {len(chunks)})")
    print(f"{'='*80}")
    print(f"Source: {chunk['source']}")
    print(f"Length: {len(chunk['text'])} characters")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"\nFull text:")
    print("-"*80)
    print(chunk['text'])
    print("-"*80)

    # Quality checks on this chunk
    has_html = '<' in chunk['text'] or '&' in chunk['text']
    has_mgmt = 'Business owner' in chunk['text']
    is_complete = len(chunk['text']) > 100

    print(f"\nQuality checks:")
    print(f"  Clean (no HTML): {'✓' if not has_html else '✗ HAS HTML'}")
    print(f"  No mgmt response: {'✓' if not has_mgmt else '✗ HAS MGMT'}")
    print(f"  Complete (>100 chars): {'✓' if is_complete else '✗ TOO SHORT'}")
    print(f"  Is complete thought: {'✓' if is_complete else '?'}")
