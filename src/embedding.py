"""
Embedding and retrieval using sentence-transformers + ChromaDB
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGSystem:
    def __init__(self, chunks_file: str = "data/chunks.json", embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model and vector store"""
        print(f"Loading chunks from {chunks_file}...")
        with open(chunks_file, 'r') as f:
            self.chunks = json.load(f)

        print(f"Initializing embedding model: {embedding_model}...")
        self.model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        self.client = chromadb.Client()
        self.collection = None

        print(f"Ready to embed {len(self.chunks)} chunks")

    def embed_and_store(self):
        """Embed all chunks and store in ChromaDB"""
        print(f"\nEmbedding {len(self.chunks)} chunks...")

        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in self.chunks]
        ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        metadatas = [
            {
                'source': chunk['source'],
                'chunk_id': str(chunk['chunk_id'])
            }
            for chunk in self.chunks
        ]

        # Embed all texts
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Create collection and add embeddings
        self.collection = self.client.create_collection(
            name="housing_reviews",
            metadata={"hnsw:space": "cosine"}
        )

        # Add documents with embeddings and metadata
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        print(f"Stored {len(self.chunks)} embeddings in ChromaDB")
        return self.collection

    def retrieve(self, query: str, top_k: int = 6) -> list:
        """
        Retrieve top-k most similar chunks for a query
        Returns list of dicts: {text, source, distance, chunk_id}
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call embed_and_store() first.")

        # Embed the query
        query_embedding = self.model.encode([query])[0]

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        retrieved = []
        if results['documents'] and len(results['documents']) > 0:
            for i, (text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                retrieved.append({
                    'rank': i + 1,
                    'text': text,
                    'source': metadata['source'],
                    'chunk_id': metadata['chunk_id'],
                    'distance': distance
                })

        return retrieved


def test_retrieval(questions: list):
    """Test retrieval on evaluation questions"""
    # Initialize system
    rag = RAGSystem()

    # Embed and store chunks
    rag.embed_and_store()

    print("\n" + "="*80)
    print("RETRIEVAL TEST")
    print("="*80)

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {question}")
        print(f"{'='*80}")

        # Retrieve
        results = rag.retrieve(question, top_k=6)

        print(f"\nRetrieved {len(results)} chunks:\n")

        for result in results:
            print(f"--- RESULT {result['rank']} ---")
            print(f"Distance: {result['distance']:.3f}")
            print(f"Source: {result['source']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Text (first 300 chars):")
            print(result['text'][:300])
            if len(result['text']) > 300:
                print("...")
            print()


if __name__ == "__main__":
    # Test questions from your evaluation plan
    test_questions = [
        "What safety issues do students report at The Circle?",
        "How quickly does Latitude management respond to resident requests?",
        "What maintenance problems do residents complain about at The Maroneal?",
        "What parking issues do students experience at District at Greenbriar?",
        "What hidden fees or lease termination charges do students report at Residences at Gramercy?"
    ]

    test_retrieval(test_questions[:3])  # Test first 3 questions
