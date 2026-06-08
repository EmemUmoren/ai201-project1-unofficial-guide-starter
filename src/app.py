"""
Gradio web interface for the Housing RAG system
"""
import gradio as gr
from generation import GroundedRAG

# Initialize the system once on startup
print("Loading RAG system...")
rag = GroundedRAG()
print("Ready!")

def answer_question(question: str) -> tuple:
    """
    Answer a user question and return answer + sources
    """
    if not question.strip():
        return "Please enter a question.", ""

    result = rag.generate_grounded_answer(question)

    answer = result['answer']
    sources_text = "\n".join([f"• {s}" for s in result['sources']])

    return answer, sources_text


# Build Gradio interface
with gr.Blocks(title="Rice Off-Campus Housing Guide") as demo:
    gr.Markdown("""
# 🏠 Rice Off-Campus Housing Guide

Ask questions about student experiences at Rice off-campus apartments.

**Examples:**
- "What do students say about safety at The Circle?"
- "How responsive is Latitude management?"
- "What maintenance issues do students report at The Maroneal?"
    """)

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What do you want to know about off-campus housing?",
                lines=3
            )
            ask_button = gr.Button("Ask", scale=1)

        with gr.Column():
            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False
            )

    sources_output = gr.Textbox(
        label="Retrieved from",
        lines=3,
        interactive=False
    )

    # Connect button and text input to function
    ask_button.click(
        answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    question_input.submit(
        answer_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    gr.Markdown("""
---
**About this system:** Answers are grounded in student reviews from Yelp and Reddit.
The system retrieves relevant reviews and uses an LLM to synthesize answers.
    """)


if __name__ == "__main__":
    demo.launch(share=False)
