import os
import faiss
import streamlit as st

from dotenv import load_dotenv
from google import genai
from PIL import Image
from retriever import load_model, load_metadata
from retriever import search, build_context

load_dotenv()

def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        raise ValueError("Gemini API key not found.")
    return genai.Client(api_key=api_key)

def generate_answer(question, context, client):

    prompt = f"""
You are answering questions about the uploaded PDF.

Use the retrieved text and images to answer the question.
Use only the provided context and do not invent information.

If the answer is not available in the context, say so.

Question:
{question}

Retrieved context:
"""
    contents = [prompt]
    for item in context:
        if item["type"] == "text":
            contents.append(
                f"Page {item['page_number'] + 1}:\n"
                f"{item['text']}"
            )
        elif item["type"] == "image":

            image = Image.open(
                item["image_path"]
            ).convert("RGB")

            contents.append(
                f"Image from page "
                f"{item['page_number'] + 1}:"
            )

            contents.append(image)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents
    )
    return response.text

def build_sources(text_results, image_results):
    sources = []
    for result in text_results:
        sources.append({
            "type": "text",
            "page_number": result["page_number"] + 1
        })

    for result in image_results:
        sources.append({
            "type": "image",
            "page_number": result["page_number"] + 1,
            "image_path": result["image_path"]
        })
    return sources


if __name__ == "__main__":

    print("Loading retriever model...")

    model, processor = load_model()

    print("Loading FAISS indexes...")

    text_index = faiss.read_index(
        "data/faiss/text_index.faiss"
    )

    image_index = faiss.read_index(
        "data/faiss/image_index.faiss"
    )

    text_metadata, image_metadata = load_metadata()

    print("Loading Gemini...")

    client = create_gemini_client()

    question = input(
        "\nEnter your question: "
    )

    print("\nSearching...")

    text_results, image_results = search(
        question,
        model,
        processor,
        text_index,
        image_index,
        text_metadata,
        image_metadata,
        top_k=5
    )

    context = build_context(
        text_results,
        image_results
    )

    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        context,
        client
    )

    sources = build_sources(
        text_results,
        image_results
    )

    print("\n" + "=" * 50)
    print("ANSWER")
    print("=" * 50)

    print(answer)

    print("\n" + "=" * 50)
    print("SOURCES")
    print("=" * 50)

    for source in sources:

        if source["type"] == "text":

            print(
                f"Text - Page "
                f"{source['page_number']}"
            )

        else:

            print(
                f"Image - Page "
                f"{source['page_number']}"
            )

            print(
                f"Path: "
                f"{source['image_path']}"
            )