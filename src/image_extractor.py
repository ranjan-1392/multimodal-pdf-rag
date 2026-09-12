import pymupdf
from pathlib import Path


def count_images_per_page(pdf_path: str) -> list[dict]:
    results = []
    doc = pymupdf.open(pdf_path)

    try:
        for page_number, page in enumerate(doc):
            images = page.get_images(full=True)

            results.append({
                "page_number": page_number,
                "image_count": len(images)
            })

    finally:
        doc.close()

    return results


def extract_images(pdf_path: str, output_dir: str) -> list[dict]:
    results = []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)

    try:
        for page_number, page in enumerate(doc):
            images = page.get_images(full=True)

            for image_number, image_info in enumerate(images):

                xref = image_info[0]

                image_data = doc.extract_image(xref)

                image_filename = (
                    f"page_{page_number}_image_{image_number}."
                    f"{image_data['ext']}"
                )

                image_path = output_path / image_filename

                with open(image_path, "wb") as image_file:
                    image_file.write(image_data["image"])

                results.append({
                    "image_id": f"page_{page_number}_image_{image_number}",
                    "page_number": page_number,
                    "xref": xref,
                    "image_path": str(image_path),
                    "width": image_data["width"],
                    "height": image_data["height"],
                    "extension": image_data["ext"]
                })

    finally:
        doc.close()

    return results


if __name__ == "__main__":
    
    pdf_path = "data/pdfs/fluid_mechanics.pdf"
    images = extract_images(pdf_path, "data/images")

    print("\ntotal images extracted:", len(images))

    tiny_images = [img for img in images if img["width"] < 50 or img["height"] < 50]

    print(f"\nTiny images (< 50x50 px): {len(tiny_images)} out of {len(images)}")
    if tiny_images:
        print("Sample tiny images:", tiny_images[:5])

    # Largest and smallest for sanity-check
    widths = [img["width"] for img in images]
    heights = [img["height"] for img in images]
    print(f"\nWidth range: {min(widths)} - {max(widths)}")
    print(f"Height range: {min(heights)} - {max(heights)}")