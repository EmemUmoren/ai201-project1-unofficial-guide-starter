"""
Embedding and retrieval using sentence-transformers + ChromaDB
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi

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

        # Initialize BM25 index on apartment names
        self._init_bm25_index()

        print(f"Ready to embed {len(self.chunks)} chunks")

    def _init_bm25_index(self):
        """Initialize BM25 index for keyword search on apartment names"""
        # Tokenize apartment names and combine with chunk text for better matching
        apartment_names = ["Circle", "Latitude", "Maroneal", "District", "Gramercy"]

        # Create documents for BM25: focus on apartment name mentions and context
        bm25_docs = []
        for chunk in self.chunks:
            # Create a document combining the chunk text with apartment name markers
            text = chunk['text'].lower()
            doc_terms = text.split()
            bm25_docs.append(doc_terms)

        self.bm25 = BM25Okapi(bm25_docs)

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

    def retrieve_hybrid(self, query: str, top_k: int = 6) -> list:
        """
        Hybrid search combining semantic (embedding) and keyword (BM25) search
        using reciprocal rank fusion (RRF).

        Args:
            query: Search query string
            top_k: Number of results to return per retrieval type

        Returns:
            List of dicts: {text, source, chunk_id, distance, hybrid_score}
            sorted by combined hybrid score (descending)
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call embed_and_store() first.")

        # Step 1: Semantic search
        query_embedding = self.model.encode([query])[0]
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Build semantic search results with ranks
        semantic_dict = {}  # chunk_index -> rank
        semantic_scores = {}  # chunk_index -> distance
        if semantic_results['documents'] and len(semantic_results['documents']) > 0:
            for rank, (text, metadata, distance) in enumerate(zip(
                semantic_results['documents'][0],
                semantic_results['metadatas'][0],
                semantic_results['distances'][0]
            )):
                chunk_idx = int(metadata['chunk_id'])
                semantic_dict[chunk_idx] = rank
                semantic_scores[chunk_idx] = distance

        # Step 2: BM25 keyword search
        query_terms = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_terms)

        # Get top-k BM25 results
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_dict = {}  # chunk_index -> rank
        for rank, idx in enumerate(bm25_indices):
            bm25_dict[int(idx)] = rank

        # Step 3: Reciprocal Rank Fusion (RRF)
        # Combine all unique chunks from both searches
        all_chunk_indices = set(semantic_dict.keys()) | set(bm25_dict.keys())

        hybrid_results = []
        for chunk_idx in all_chunk_indices:
            chunk = self.chunks[chunk_idx]

            # RRF formula: score = 1/(rank + 1)
            semantic_rank = semantic_dict.get(chunk_idx, top_k)  # Default to worst rank if not found
            bm25_rank = bm25_dict.get(chunk_idx, top_k)

            semantic_contrib = 1.0 / (semantic_rank + 1)
            bm25_contrib = 1.0 / (bm25_rank + 1)

            # Combined score as sum of normalized contributions
            hybrid_score = semantic_contrib + bm25_contrib

            # Use distance from semantic search if available, else use BM25 rank
            distance = semantic_scores.get(chunk_idx, float(bm25_rank))

            hybrid_results.append({
                'text': chunk['text'],
                'source': chunk['source'],
                'chunk_id': str(chunk['chunk_id']),
                'distance': distance,
                'hybrid_score': hybrid_score,
                'semantic_rank': semantic_rank if chunk_idx in semantic_dict else None,
                'bm25_rank': bm25_rank if chunk_idx in bm25_dict else None
            })

        # Sort by hybrid score (descending) and return top-k
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Add rank field and return top-k
        for rank, result in enumerate(hybrid_results[:top_k], 1):
            result['rank'] = rank

        return hybrid_results[:top_k]


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
