import pymupdf

def extract_text_per_page(pdf_path: str) -> list[dict]:
    pages = []
    doc = pymupdf.open(pdf_path)
    try:
        for page_number, page in enumerate(doc):
            text = page.get_text()
            pages.append({
                "page_number": page_number,
                "text": text
            })
    finally:
        doc.close()
    return pages

# ---- Everything below this line is ONLY for manual testing/debugging ----

if __name__ == "__main__":
    pdf_path = "data/pdfs/fluid_mechanics.pdf"
    pages = extract_text_per_page(pdf_path)

    print("total_pages:", len(pages))
    print("\nFirst page:")
    print(pages[0])

    # --- Diagnostic checks ---
    empty_pages = [p["page_number"] for p in pages if not p["text"].strip()]
    print(f"\nPages with NO extractable text: {len(empty_pages)} out of {len(pages)}")

    if empty_pages:
        print("Empty page numbers (first 20):", empty_pages[:20])

    non_empty_count = len(pages) - len(empty_pages)
    print(f"Pages WITH text: {non_empty_count} ({non_empty_count/len(pages)*100:.1f}%)")

    print("\nPage 50 text preview:")
    print(pages[50]["text"][:300])
    
    
        
        
        
        
        
    