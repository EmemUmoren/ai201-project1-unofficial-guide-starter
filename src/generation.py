"""
Grounded response generation using Groq LLM
"""
import os
from dotenv import load_dotenv
from groq import Groq
from embedding import RAGSystem

load_dotenv()

class GroundedRAG:
    def __init__(self):
        """Initialize Groq client and RAG system"""
        self.groq_key = os.getenv("GROQ_API_KEY")
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.client = Groq(api_key=self.groq_key)
        self.rag = RAGSystem()
        self.rag.embed_and_store()

    def generate_grounded_answer(self, query: str, top_k: int = 6) -> dict:
        """
        Generate a grounded response using only retrieved context
        Returns: {answer, sources, retrieved_chunks}
        """
        # Retrieve relevant chunks
        retrieved = self.rag.retrieve(query, top_k=top_k)

        if not retrieved:
            return {
                'answer': "I don't have enough information to answer that question.",
                'sources': [],
                'retrieved_chunks': []
            }

        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in retrieved
        ])

        # System prompt that enforces grounding
        system_prompt = """You are a helpful assistant that answers questions about off-campus student housing experiences at Rice University.

IMPORTANT: You must answer using ONLY the information provided in the retrieved documents below. Do not use any information from your training data or general knowledge.

If the documents don't contain enough information to answer the question, say: "I don't have enough information about that based on the available reviews."

Always cite which document(s) your answer comes from."""

        # User prompt with context
        user_prompt = f"""Based on the following student reviews and housing discussions:

{context}

Please answer this question:
{query}

Remember: Answer only using the information above. Do not make up information."""

        # Generate response with Groq
        message = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = message.choices[0].message.content

        # Extract unique sources from retrieved chunks
        sources = list(set([chunk['source'] for chunk in retrieved]))

        return {
            'answer': answer,
            'sources': sources,
            'retrieved_chunks': retrieved
        }


def test_all_questions():
    """Test generation on all 5 evaluation questions"""
    print("Initializing Grounded RAG system...")
    rag = GroundedRAG()

    questions = [
        "What safety issues do students report at The Circle?",
        "How quickly does Latitude management respond to resident requests?",
        "What maintenance problems do residents complain about at The Maroneal?",
        "What parking issues do students experience at District at Greenbriar?",
        "What hidden fees or lease termination charges do students report at Residences at Gramercy?"
    ]

    print("\n" + "="*80)
    print("GENERATION TEST: All 5 Evaluation Questions")
    print("="*80)

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Q{i}: {question}")
        print(f"{'='*80}\n")

        result = rag.generate_grounded_answer(question)

        print("ANSWER:")
        print(result['answer'])
        print(f"\nSOURCES: {', '.join(result['sources'])}")
        print(f"\nRETRIEVED {len(result['retrieved_chunks'])} chunks:")
        for j, chunk in enumerate(result['retrieved_chunks'][:3], 1):  # Show top 3
            print(f"  {j}. {chunk['source']} (distance: {chunk['distance']:.3f})")


if __name__ == "__main__":
    test_all_questions()
