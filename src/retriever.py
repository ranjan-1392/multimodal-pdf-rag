import faiss
import json
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

EMBEDDINGS_DIR = Path("data/embeddings")
INDEX_DIR = Path("data/faiss")
MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SCORE_THRESHOLD = 0.27
CANDIDATE_POOL_SIZE = 15
MAX_IMAGES_HARD_CAP = 8

def load_model():

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print("CLIP loaded successfully.")
    return model, processor


def load_metadata():

    with open(
        EMBEDDINGS_DIR / "text_metadata.json",
        encoding="utf-8"
    ) as f:
        text_metadata = json.load(f)

    with open(
        EMBEDDINGS_DIR / "image_metadata.json",
        encoding="utf-8"
    ) as f:
        image_metadata = json.load(f)

    return text_metadata, image_metadata


def search(
    query,
    model,
    processor,
    text_index,
    image_index,
    text_metadata,
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

    query_vector = (
        outputs.pooler_output
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    faiss.normalize_L2(query_vector)


    # -------------------------
    # Text retrieval
    # -------------------------

    text_scores, text_ids = text_index.search(query_vector, top_k)

    text_results = []

    for score, idx in zip(text_scores[0], text_ids[0]):
        if idx == -1:
            continue
        result = text_metadata[idx].copy()
        result["score"] = float(score)
        text_results.append(result)

    # -------------------------
    # Image retrieval
    # -------------------------

    image_scores, image_ids = image_index.search(query_vector, CANDIDATE_POOL_SIZE)

    image_results = []

    print("\nImage retrieval scores:")

    for score, idx in zip(image_scores[0], image_ids[0]):
        if idx == -1:
            continue
        page_number = (image_metadata[idx]["page_number"] + 1)
        print(f"Page {page_number} -> {score:.4f}")
        
        if image_metadata[idx]["page_number"] == 0:
            print("  -> Image filtered out "
                   "(book cover)")
            continue

        if score < IMAGE_SCORE_THRESHOLD:

            print(
                "  -> Image filtered out "
                "(below threshold)"
            )

            continue

        result = image_metadata[idx].copy()
        result["score"] = float(score)
        image_results.append(result)
        print("  -> Image kept")

    # Keep only the best few images
    # after applying the similarity threshold.

    image_results = image_results[:MAX_IMAGES_HARD_CAP]


    print(
        f"\nFinal images after "
        f"{len(image_results)}"
    )
    return text_results, image_results


def build_context(text_results, image_results):
    context = []

    # Add retrieved text

    for result in text_results:
        context.append({
            "type": "text",
            "page_number": result["page_number"],
            "text": result["text"],
            "score": result["score"]
        })

    # Add filtered image results

    for result in image_results:
        context.append({
            "type": "image",
            "page_number": result["page_number"],
            "image_path": result["image_path"],
            "score": result["score"]
        })
    return context