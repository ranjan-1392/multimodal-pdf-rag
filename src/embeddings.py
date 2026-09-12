from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import json
from pathlib import Path
import torch


MODEL_NAME = "openai/clip-vit-base-patch32"

EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def load_clip_model():
    print("Loading CLIP model...")

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    model.eval()

    print("CLIP loaded successfully.")

    return model, processor


def embed_texts(texts: list[str], model, processor, batch_size: int = 32) -> np.ndarray:
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        embeddings = outputs.pooler_output

        all_embeddings.append(embeddings.cpu().numpy())

        print(f"Text embeddings: "f"{min(start + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings)


def embed_images(images: list[dict], model, processor, batch_size: int = 16) -> np.ndarray:

    all_embeddings = []

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        pil_images = []

        for image_info in batch:
            image = Image.open(image_info["image_path"]).convert("RGB")
            pil_images.append(image)  

        inputs = processor(images=pil_images, return_tensors="pt")
      
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        embeddings = outputs.pooler_output

        all_embeddings.append(embeddings.cpu().numpy())
      
        print(f"Image embeddings: "f"{min(start + batch_size, len(images))}/{len(images)}")
 
    return np.vstack(all_embeddings)


if __name__ == "__main__":

    from pdf_parser import extract_text_per_page
    from image_extractor import extract_images
    from text_chunker import build_text_chunks

    pdf_path = "data/pdfs/fluid_mechanics.pdf"

    # Load PDF data

    print("\nReading PDF...")
    pages = extract_text_per_page(pdf_path)
    print("Building text chunks...")
    text_chunks = build_text_chunks(pages)
    print(f"Total text chunks: {len(text_chunks)}")
    print("\nExtracting images...")

    images = extract_images(pdf_path, "data/images")

    print(f"Total images: {len(images)}")

    # Load CLIP

    model, processor = load_clip_model()

    # Text embeddings
    print("\nGenerating text embeddings...")
    texts = [chunk.text for chunk in text_chunks]
    text_embeddings = embed_texts(texts, model, processor)
    print("\nText embedding matrix:",text_embeddings.shape)

    # Save text embeddings

    np.save(EMBEDDINGS_DIR / "text_embeddings.npy", text_embeddings)

    text_metadata = [
        {
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "content_type": chunk.content_type,
            "text": chunk.text
        }
        for chunk in text_chunks
    ]

    with open(EMBEDDINGS_DIR / "text_metadata.json", "w", encoding="utf-8") as file:
        json.dump(text_metadata, file, ensure_ascii=False, indent=2)
    print("Text embeddings saved.")

    # Image embeddings

    print("\nGenerating image embeddings...")
    image_embeddings = embed_images(images, model, processor)
    print("\nImage embedding matrix:", image_embeddings.shape)

    # Save image embeddings

    np.save(EMBEDDINGS_DIR / "image_embeddings.npy",image_embeddings)

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

    with open(EMBEDDINGS_DIR / "image_metadata.json", "w", encoding="utf-8") as file:

        json.dump(image_metadata, file, ensure_ascii=False, indent=2)

    print("Image embeddings saved.")
    print("\nM6.2 complete!")