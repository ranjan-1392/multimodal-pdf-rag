import faiss
import json
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

EMBEDDINGS_DIR = Path("data/embeddings")
INDEX_DIR = Path("data/faiss")
MODEL_NAME = "openai/clip-vit-base-patch32"

def load_model():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def load_metadata():
    with open(
        EMBEDDINGS_DIR / "image_metadata.json",
        encoding="utf-8"
    ) as f:
        image_metadata = json.load(f)
    return image_metadata


def search_images(
    query,
    model,
    processor,
    image_index,
    image_metadata,
    top_k=5
):

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model.get_text_features(**inputs)
    query_vector = outputs.pooler_output.detach().cpu().numpy()
    query_vector = query_vector.astype("float32")

    faiss.normalize_L2(query_vector)

    scores, ids = image_index.search(query_vector,top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        result = image_metadata[idx].copy()
        result["score"] = float(score)
        results.append(result)
    return results


if __name__ == "__main__":
    query = input("Enter image query: ")
    print("\nLoading model...")
    model, processor = load_model()
    print("Loading FAISS index...")
    image_index = faiss.read_index(str(INDEX_DIR / "image_index.faiss"))
    image_metadata = load_metadata()

    results = search_images(
        query,
        model,
        processor,
        image_index,
        image_metadata,
        top_k=5
    )

    print("\n" + "=" * 60)
    print("TOP IMAGE RESULTS")
    print("=" * 60)

    for i, result in enumerate(results, start=1):

        print(f"\n{i}. Page: {result['page_number'] + 1}")
        print(f"Score: {result['score']:.4f}")
        print(f"Image: {result['image_path']}")
        