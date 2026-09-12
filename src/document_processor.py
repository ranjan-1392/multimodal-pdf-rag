from pathlib import Path

from pdf_parser import extract_text_per_page
from image_extractor import extract_images
from text_chunker import build_text_chunks
from embeddings import load_clip_model
from embeddings import embed_texts, embed_images
from vector_store import create_index
from vector_store import save_processed_data


def process_pdf(
    pdf_path,
    model=None,
    processor=None,
    cache_dir=None
):

    print("Reading PDF...")
    pages = extract_text_per_page(pdf_path)
    print(f"Total pages: {len(pages)}")
    print("\nBuilding text chunks...")
    text_chunks = build_text_chunks(pages)

    print(f"Total text chunks: {len(text_chunks)}")

    if cache_dir is not None:
        image_output_dir = (Path(cache_dir) / "images")
    else:
        image_output_dir = (Path("data/uploads") /f"{Path(pdf_path).stem}_images")

    print("\nExtracting images...")

    images = extract_images(pdf_path, str(image_output_dir))
    print(f"Total images: {len(images)}")


    if model is None or processor is None:
        print("\nLoading CLIP model...")
        model, processor = load_clip_model()


    print("\nGenerating text embeddings...")

    texts = [
        chunk.text
        for chunk in text_chunks
    ]

    text_embeddings = embed_texts(
        texts,
        model,
        processor
    )

    print(
        "Text embedding matrix:",
        text_embeddings.shape
    )


    print("\nGenerating image embeddings...")

    image_embeddings = embed_images(
        images,
        model,
        processor
    )

    print(
        "Image embedding matrix:",
        image_embeddings.shape
    )


    print("\nCreating FAISS indexes...")

    text_index = create_index(
        text_embeddings.astype("float32")
    )

    image_index = create_index(
        image_embeddings.astype("float32")
    )


    text_metadata = [

        {
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "content_type": chunk.content_type,
            "text": chunk.text
        }

        for chunk in text_chunks
    ]


    image_metadata = [

        {
            "image_id": image["image_id"],
            "page_number": image["page_number"],
            "image_path": image["image_path"],
            "width": image["width"],
            "height": image["height"]
        }

        for image in images
    ]


    if cache_dir is not None:

        print("\nSaving processed data...")

        save_processed_data(
            cache_dir,
            text_index,
            image_index,
            text_metadata,
            image_metadata
        )


    print("\nPDF processing complete!")


    return (
        model,
        processor,
        text_index,
        image_index,
        text_metadata,
        image_metadata
    )