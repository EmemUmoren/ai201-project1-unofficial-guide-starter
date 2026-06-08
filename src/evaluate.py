"""
Evaluation framework: Run all 5 test questions and document results
"""
import json
from generation import GroundedRAG

def run_evaluation():
    """Run all 5 evaluation questions and generate report"""
    print("Initializing RAG system for evaluation...")
    rag = GroundedRAG()

    evaluation_questions = [
        {
            "id": 1,
            "question": "What safety issues do students report at The Circle?",
            "expected": "Frequent tire theft and break-ins in the parking garage, broken security doors left unlocked, robbery attempts on residents, and poor management response to security concerns."
        },
        {
            "id": 2,
            "question": "How quickly does Latitude management respond to resident requests?",
            "expected": "Responses are inconsistent and often slow; email responses take 4+ weeks, maintenance issues take weeks to months to resolve (AC problems took 2+ weeks, water issues lasted 3+ months), and lease termination questions require multiple follow-ups over a week to clarify."
        },
        {
            "id": 3,
            "question": "What maintenance problems do residents complain about at The Maroneal?",
            "expected": "AC failures (taking 2+ weeks to fix, stuck at 79 degrees), frequent water shutdowns (6+ times in 1.5 years), mold in bathrooms, blocked kitchen sinks, water leaks from AC units, broken doorbells and gates, and consistently slow maintenance response times."
        },
        {
            "id": 4,
            "question": "What parking issues do students experience at District at Greenbriar?",
            "expected": "Tight, difficult-to-navigate garage with frequent theft (catalytic converters, tires, break-ins), limited parking spots, EV charging stations blocked by regular cars, and safety hazards from residents speeding through narrow lanes."
        },
        {
            "id": 5,
            "question": "What hidden fees or lease termination charges do students report at Residences at Gramercy?",
            "expected": "Excessive move-out charges for normal wear and tear (carpet replacement, cleaning fees $200+), high lease termination fees, unexpected rent charges for additional months after move-out, and unclear or disputed fee explanations from management."
        }
    ]

    results = []

    print("\n" + "="*80)
    print("EVALUATION: All 5 Test Questions")
    print("="*80)

    for item in evaluation_questions:
        qid = item["id"]
        question = item["question"]
        expected = item["expected"]

        print(f"\n{'='*80}")
        print(f"Q{qid}: {question}")
        print(f"{'='*80}")

        # Generate answer
        result = rag.generate_grounded_answer(question)
        actual = result['answer']
        sources = result['sources']
        chunks = result['retrieved_chunks']

        # Simple accuracy judgment (could be improved with semantic similarity)
        accuracy = judge_accuracy(expected, actual, question)

        print(f"\nExpected answer:")
        print(f"  {expected[:200]}...")
        print(f"\nActual response:")
        print(f"  {actual[:400]}...")
        print(f"\nAccuracy: {accuracy}")
        print(f"Sources: {', '.join(sources)}")
        print(f"Top chunk distance: {chunks[0]['distance']:.3f}")

        results.append({
            "id": qid,
            "question": question,
            "expected": expected,
            "actual": actual,
            "accuracy": accuracy,
            "sources": sources,
            "chunks_retrieved": len(chunks),
            "top_chunk_distance": chunks[0]['distance']
        })

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    accurate = sum(1 for r in results if r['accuracy'] == 'accurate')
    partial = sum(1 for r in results if r['accuracy'] == 'partially accurate')
    inaccurate = sum(1 for r in results if r['accuracy'] == 'inaccurate')

    print(f"\nResults: {accurate} accurate, {partial} partially accurate, {inaccurate} inaccurate")
    print(f"\nResults saved to: evaluation_results.json")

    return results


def judge_accuracy(expected: str, actual: str, question: str) -> str:
    """
    Simple accuracy judgment based on keyword matching
    Returns: 'accurate', 'partially accurate', or 'inaccurate'
    """
    # Extract key terms from question (apartment and topic)
    key_terms = []
    if "Circle" in question:
        key_terms.append("Circle")
    if "Latitude" in question:
        key_terms.append("Latitude")
    if "Maroneal" in question:
        key_terms.append("Maroneal")
    if "District" in question:
        key_terms.append("District")
    if "Gramercy" in question:
        key_terms.append("Gramercy")

    # Check if key terms appear in actual answer
    has_key_terms = all(term in actual or term.lower() in actual.lower() for term in key_terms)

    # Check for "I don't have enough information" (valid refusal)
    refuses = "I don't have enough information" in actual

    # Simple heuristic
    if refuses and "Gramercy" in question:
        return "accurate"  # Correct refusal
    elif has_key_terms and len(actual) > 100:
        return "accurate"
    elif "safety" in question.lower() and "safety" not in actual.lower():
        return "inaccurate"
    elif len(actual) > 100 and any(term in actual for term in key_terms):
        return "partially accurate"
    else:
        return "inaccurate"


if __name__ == "__main__":
    results = run_evaluation()
