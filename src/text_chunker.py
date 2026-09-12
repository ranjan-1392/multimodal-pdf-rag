from schema import DocumentChunk

def chunk_text(text:str, chunk_size:int=1000, overlap:int=150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
                
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
            
        if end >= len(text):
            break
            
        next_start = end - overlap
        
        if next_start <= start:
            next_start = end
            
        start = next_start

        
    return chunks


# Page information preserve
def build_text_chunks(pages: list[dict], chunk_size: int = 1000, overlap: int = 150) -> list[DocumentChunk]:
    chunks = []
    
    for page in pages:
        page_number = page["page_number"]
        text = page["text"]
        
        if not text.strip():
            continue
        
        text_parts = chunk_text(text, chunk_size, overlap)
        
        
        for chunk_number, text_part in enumerate(text_parts):
            chunk = DocumentChunk(
                chunk_id=f"text_page_{page_number}_chunk_{chunk_number}",
                page_number=page_number,
                content_type="text",
                text=text_part
            )
            
            chunks.append(chunk)
            
        if page_number % 100 == 0:
            print(
                f"Processed page {page_number} | "
                f"Total chunks: {len(chunks)}")
            
    return chunks
                        
                        
                    
         
    


if __name__ == "__main__":

    from pdf_parser import extract_text_per_page

    pdf_path = "data/pdfs/fluid_mechanics.pdf"

    print("Reading PDF...")

    pages = extract_text_per_page(pdf_path)

    print("PDF reading complete.")
    print("Pages:", len(pages))

    print("Starting chunking...")

    chunks = build_text_chunks(pages)

    print("Chunking complete.")

    print("Total text chunks:", len(chunks))

    print("\nFirst 5 chunks:")

    for chunk in chunks[:5]:
        print("\n", chunk)