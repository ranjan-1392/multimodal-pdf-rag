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

🔍 How It Works
1. PDF Processing

The application accepts a PDF uploaded through the Streamlit interface.

PyMuPDF is used to:

extract text page-by-page
extract embedded images
preserve page-level metadata
2. Text Chunking

Extracted text is divided into smaller overlapping chunks before generating embeddings.

Current configuration:

Chunk size: 1000 characters
Overlap:    150 characters

Each text chunk retains its original PDF page number so retrieved information can be traced back to its source.

3. Multimodal Embeddings

The project uses:

openai/clip-vit-base-patch32

CLIP generates embeddings for:

text
images

Both embedding types have a dimensionality of:

512

The text and image embeddings are used for similarity-based retrieval.

4. Vector Search

FAISS is used as the vector search engine.

The current implementation maintains separate indexes for text and images:

                    Query
                      │
              CLIP Text Encoder
                      │
                Query Embedding
                 ┌────┴────┐
                 ▼         ▼
            Text FAISS   Image FAISS
                 │         │
                 ▼         ▼
          Text Results  Image Results

Embeddings are normalized before indexing, allowing inner-product search to represent cosine similarity.

5. Multimodal Retrieval

For each user query, the system:

Generates a CLIP embedding for the query.
Searches the text FAISS index.
Searches the image FAISS index.
Applies relevance filtering to image results.
Selects the retrieved text and images.
Builds a multimodal context for the language model.
6. Gemini Answer Generation

The retrieved context is passed to Gemini along with the user's question.

The generation pipeline is instructed to:

use the retrieved context
avoid inventing information
state when the answer is not available in the retrieved context

Images retrieved from the PDF can also be passed to Gemini as visual context.

7. Source Citations

Each text chunk and image maintains its original PDF page number.

The application displays the source pages associated with the retrieved content so users can trace the answer back to the document.

📊 Current V1 Statistics

The initial system was tested using a technical fluid mechanics textbook containing 899 PDF pages.

Metric	Value
PDF Pages	899
Pages with Extractable Text	898
Text Chunks	2,971
Extracted Image Occurrences	585
CLIP Embedding Dimension	512
Text Chunk Size	1,000 characters
Chunk Overlap	150 characters

Note: The 585 image count represents extracted image occurrences. Some underlying PDF image resources may appear more than once in the document.

🛠️ Tech Stack
Component	Technology
Language	Python
PDF Processing	PyMuPDF
Text Chunking	Custom Python
Multimodal Embeddings	OpenAI CLIP
Vector Search	FAISS
Image Processing	Pillow
LLM	Google Gemini
UI	Streamlit
Deep Learning	PyTorch
Model Library	Hugging Face Transformers
Numerical Processing	NumPy
Environment Management	python-dotenv
📁 Project Structure
multimodal-pdf-rag/
│
├── src/
│   ├── app.py                  # Streamlit application
│   ├── pdf_parser.py           # PDF text extraction
│   ├── image_extractor.py      # PDF image extraction
│   ├── schema.py               # DocumentChunk data structure
│   ├── metadata_builder.py     # Document metadata creation
│   ├── text_chunker.py         # Text chunking
│   ├── embeddings.py           # CLIP text/image embeddings
│   ├── vector_store.py         # FAISS indexes and caching
│   ├── retriever.py            # Multimodal retrieval
│   ├── rag_pipeline.py         # Gemini generation pipeline
│   ├── document_processor.py   # End-to-end document processing
│   └── test_image_retrieval.py # Image retrieval experiment
│
├── data/
│   ├── pdfs/                   # Local PDFs (ignored by Git)
│   ├── uploads/                # Uploaded PDFs (ignored by Git)
│   ├── images/                 # Extracted images (ignored by Git)
│   ├── embeddings/             # Generated embeddings (ignored by Git)
│   ├── faiss/                  # FAISS indexes (ignored by Git)
│   └── cache/                  # Processed document cache
│
├── .env                        # Local API key (ignored by Git)
├── .gitignore
├── requirements.txt
└── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/ranjan-1392/multimodal-pdf-rag.git
cd multimodal-pdf-rag
2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
🔑 Environment Setup

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key

The .env file is excluded from Git using .gitignore.

🚀 Run the Application

Start the Streamlit application:

streamlit run src/app.py

The application will open in your browser.

Application Flow
Upload PDF
    ↓
Process Document
    ↓
Extract Text + Images
    ↓
Create CLIP Embeddings
    ↓
Build FAISS Indexes
    ↓
Ask Question
    ↓
Retrieve Relevant Text + Images
    ↓
Generate Gemini Answer
    ↓
Display Answer + Sources
⚡ Document Processing Cache

Processing a PDF requires:

text extraction
image extraction
CLIP embedding generation
FAISS index creation

To avoid repeating this work for the same document, the application generates a SHA-256 hash of the uploaded PDF and uses it as a cache identifier.

PDF
 │
 ▼
SHA-256 Hash
 │
 ▼
data/cache/<document_hash>/
 │
 ├── FAISS indexes
 └── metadata

When the same document is uploaded again, the cached processed data can be loaded instead of rebuilding the complete pipeline.

🧪 Example Questions

Example questions that can be asked about a technical PDF:

What is the difference between laminar and turbulent flow?

What equation is used to calculate the Reynolds number?

Explain the concept shown in the relevant diagram.

What does this figure represent?

Explain the relationship between pressure and velocity shown in the retrieved content.

The quality of visual questions depends on the relevance of the images retrieved by the CLIP-based image search.

⚠️ Current Limitations

This is the V1 implementation and there are several areas that can be improved.

Image Retrieval

Image retrieval currently relies on CLIP similarity and may sometimes retrieve visually or semantically related images that are not the most relevant figure for a particular technical question.

Text Embeddings

CLIP was designed primarily for image-text alignment rather than long-form technical document retrieval.

A dedicated text embedding model could potentially improve retrieval quality for technical questions.

Tables

Table extraction and table retrieval are not currently part of the V1 pipeline.

Separate Text and Image Indexes

Text and image retrieval currently use separate FAISS indexes rather than a single unified multimodal index.

Formal Evaluation

V1 does not currently include a formal retrieval or answer-quality evaluation benchmark.

🔮 Future Improvements

Potential improvements for future versions include:

Better semantic text embedding models
Improved page-aware image retrieval
Hybrid text and image retrieval
Retrieval reranking
Better handling of technical figures and diagrams
Table extraction and retrieval
Automated retrieval evaluation
Answer-quality evaluation
Improved multimodal context selection
Deployment and performance improvements
📌 V1 → V2

The main goal of V1 was to build and understand the complete multimodal RAG pipeline:

PDF
 ↓
Text + Image Extraction
 ↓
Chunking + Metadata
 ↓
CLIP Embeddings
 ↓
FAISS Retrieval
 ↓
Gemini
 ↓
Streamlit

Future versions can focus on improving retrieval quality, evaluation, and multimodal grounding rather than simply adding more components.

👨‍💻 Author

Prabhat Ranjan

GitHub: @ranjan-1392

📄 License

This project is intended for educational and portfolio purposes.

⭐ If you found this project useful, consider starring the repository.
