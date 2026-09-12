import os
import hashlib
import streamlit as st

from document_processor import process_pdf
from retriever import search
from retriever import build_context
from rag_pipeline import create_gemini_client
from rag_pipeline import generate_answer
from rag_pipeline import build_sources
from embeddings import load_clip_model
from vector_store import cache_exists
from vector_store import load_processed_data


st.set_page_config(
    page_title="Multimodal PDF RAG",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def get_clip_model():
    return load_clip_model()

def get_pdf_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()
        
st.title("📚 Multimodal PDF RAG")
st.write("Upload a PDF and ask questions using its text and images.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    st.write(f"Selected file: **{uploaded_file.name}**")
    file_bytes = uploaded_file.getvalue()
    pdf_hash = get_pdf_hash(file_bytes)

    cache_dir = os.path.join(
        "data",
        "cache",
        pdf_hash
    )


    if st.session_state.get("pdf_hash") != pdf_hash:

        st.session_state.pdf_ready = False


    if st.button(
        "Process PDF",
        type="primary"
    ):

        if cache_exists(cache_dir):

            with st.spinner(
                "Loading processed PDF..."
            ):

                model, processor = get_clip_model()

                (
                    text_index,
                    image_index,
                    text_metadata,
                    image_metadata
                ) = load_processed_data(
                    cache_dir
                )

            st.info(
                "Cached data found. "
                "No re-processing was required."
            )


        else:

            with st.spinner(
                "Processing PDF... This may take some time."
            ):

                upload_dir = "data/uploads"

                os.makedirs(
                    upload_dir,
                    exist_ok=True
                )


                pdf_path = os.path.join(
                    upload_dir,
                    uploaded_file.name
                )


                with open(
                    pdf_path,
                    "wb"
                ) as file:

                    file.write(file_bytes)


                model, processor = get_clip_model()


                (
                    model,
                    processor,
                    text_index,
                    image_index,
                    text_metadata,
                    image_metadata
                ) = process_pdf(
                    pdf_path,
                    model,
                    processor,
                    cache_dir
                )


            st.success(
                "PDF processed successfully!"
            )


        st.session_state.model = model
        st.session_state.processor = processor

        st.session_state.text_index = text_index
        st.session_state.image_index = image_index

        st.session_state.text_metadata = text_metadata
        st.session_state.image_metadata = image_metadata

        st.session_state.pdf_hash = pdf_hash
        st.session_state.pdf_ready = True


if st.session_state.get(
    "pdf_ready",
    False
):

    st.divider()


    question = st.text_input(
        "Ask a question about the uploaded PDF",
        placeholder="Example: What is a hydraulic jump?"
    )


    if st.button("🔍 Ask"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )


        else:

            with st.spinner(
                "Searching the PDF and generating answer..."
            ):

                model = (
                    st.session_state.model
                )

                processor = (
                    st.session_state.processor
                )

                text_index = (
                    st.session_state.text_index
                )

                image_index = (
                    st.session_state.image_index
                )

                text_metadata = (
                    st.session_state.text_metadata
                )

                image_metadata = (
                    st.session_state.image_metadata
                )


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


                client = create_gemini_client()


                answer = generate_answer(
                    question,
                    context,
                    client
                )


                sources = build_sources(
                    text_results,
                    image_results
                )


            st.divider()


            st.subheader("💡 Answer")

            st.write(answer)


            st.subheader("📚 Sources")


            text_pages = set()

            image_sources = []


            for source in sources:

                if source["type"] == "text":

                    text_pages.add(
                        source["page_number"]
                    )

                elif source["type"] == "image":

                    image_sources.append(
                        source
                    )


            if text_pages:

                st.write("**Text sources:**")

                for page in sorted(text_pages):

                    st.write(
                        f"📄 Page {page}"
                    )


            if image_sources:

                st.write(
                    "**Retrieved images:**"
                )


                columns = st.columns(
                    min(
                        len(image_sources),
                        3
                    )
                )


                for i, source in enumerate(
                    image_sources
                ):

                    with columns[
                        i % 3
                    ]:

                        st.caption(
                            f"🖼️ Page "
                            f"{source['page_number']}"
                        )


                        st.image(
                            source["image_path"],
                            width="stretch"
                        )