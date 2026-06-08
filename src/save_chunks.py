"""
Save cleaned chunks to a file for next milestone (embedding)
"""
import json
from ingestion import load_documents, chunk_documents

documents = load_documents()
chunks = chunk_documents(documents)

# Save chunks as JSON
output_file = "data/chunks.json"

import os
os.makedirs("data", exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(chunks, f, indent=2)

print(f"Saved {len(chunks)} chunks to {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")
