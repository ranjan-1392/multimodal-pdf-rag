from schema import DocumentChunk

def build_text_chunks(pages: list[dict]) -> list[DocumentChunk]:
    chunks = []
    
    for page in pages:
        if not page["text"].strip():
            continue
        
        chunk = DocumentChunk(
            chunk_id=f"text_page_{page['page_number']}",
            page_number=page["page_number"],
            content_type="text",
            text=page["text"]
        )

        chunks.append(chunk)
        
    return chunks


def build_image_chunks(images: list[dict]) -> list[DocumentChunk]:
    chunks = []

    for image in images:

        chunk = DocumentChunk(
            chunk_id=image["image_id"],
            page_number=image["page_number"],
            content_type="image",
            image_path=image["image_path"]
        )

        chunks.append(chunk)

    return chunks


def build_document_chunks(pages: list[dict], images: list[dict]) -> list[DocumentChunk]:
    
    text_chunks = build_text_chunks(pages)
    image_chunks = build_image_chunks(images)
    
    return text_chunks + image_chunks
    
    

if __name__ == "__main__":
    from pdf_parser import extract_text_per_page
    from image_extractor import extract_images

    pdf_path = "data/pdfs/fluid_mechanics.pdf"

    pages = extract_text_per_page(pdf_path)

    images = extract_images(
        pdf_path,
        "data/images"
    )

    chunks = build_document_chunks(
        pages,
        images
    )

    print("Total DocumentChunks:", len(chunks))

    print("\nFirst 3 chunks:")

    for chunk in chunks[:3]:
        print(chunk)

    print("\nLast 3 chunks:")

    for chunk in chunks[-3:]:
        print(chunk)
    