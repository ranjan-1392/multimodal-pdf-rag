# Multimodal PDF RAG

> A multimodal Retrieval-Augmented Generation system for asking questions about technical PDF documents using both text and images, powered by CLIP, FAISS, Gemini, and Streamlit.

---

## 🎬 Demo

> Demo screenshot / GIF coming soon.

The application provides a Streamlit interface where users can upload a PDF, ask questions, and receive answers along with the relevant source pages and retrieved images.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 PDF Upload | Upload different PDF documents directly through the Streamlit interface |
| 📝 Text Extraction | Extracts text page-by-page using PyMuPDF |
| 🖼️ Image Extraction | Extracts embedded images from PDF documents |
| ✂️ Text Chunking | Splits page text into overlapping chunks for better retrieval |
| 🔤 Multimodal Embeddings | Generates embeddings for both text and images using CLIP |
| 🔎 Semantic Search | Uses FAISS for similarity-based vector retrieval |
| 🧠 Multimodal Retrieval | Retrieves relevant text and image context for a user query |
| 🤖 Gemini Generation | Generates answers using retrieved text and image context |
| 📚 Source Citations | Displays the PDF pages associated with retrieved content |
| ⚡ Document Caching | Caches processed documents locally to avoid repeated processing |
| 📤 Dynamic Documents | Supports processing different uploaded PDFs |

---

## 🏗️ Architecture

```text
                         PDF Document
                              │
                              ▼
                   ┌─────────────────────┐
                   │    Document Parser   │
                   │       PyMuPDF        │
                   └──────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
               Text Pages          PDF Images
                    │                   │
                    ▼                   ▼
              Text Chunking       Image Metadata
                    │                   │
                    ▼                   ▼
             CLIP Text Encoder   CLIP Image Encoder
                    │                   │
                    ▼                   ▼
             Text Embeddings     Image Embeddings
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     ┌────────────────┐
                     │      FAISS     │
                     │ Vector Search  │
                     └───────┬────────┘
                             │
                             ▼
                    Multimodal Retrieval
                       ┌─────┴─────┐
                       ▼           ▼
                  Text Results  Image Results
                       │           │
                       └─────┬─────┘
                             ▼
                    Retrieved Context
                             │
                             ▼
                      Gemini 3.6 Flash
                             │
                             ▼
                    Answer + Page Sources
                             │
                             ▼
                       Streamlit UI
## How It works
