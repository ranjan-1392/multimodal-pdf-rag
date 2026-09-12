import faiss
import numpy as np
import json
from pathlib import Path

EMBEDDINGS_DIR = Path("data/embeddings")
INDEX_DIR = Path("data/faiss")

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    text_vectors = np.load(EMBEDDINGS_DIR / "text_embeddings.npy").astype("float32")
    image_vectors = np.load(EMBEDDINGS_DIR / "image_embeddings.npy").astype("float32")

    with open(EMBEDDINGS_DIR / "text_metadata.json", encoding="utf-8") as f:
        text_metadata = json.load(f)

    with open(EMBEDDINGS_DIR / "image_metadata.json", encoding="utf-8") as f:
        image_metadata = json.load(f)
 
    return (text_vectors, image_vectors, text_metadata, image_metadata)

def create_index(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def save_processed_data(
    cache_dir,
    text_index,
    image_index,
    text_metadata,
    image_metadata
):

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(text_index, str(cache_path / "text_index.faiss"))
    faiss.write_index(image_index, str(cache_path / "image_index.faiss"))

    with open(
        cache_path / "text_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            text_metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        cache_path / "image_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            image_metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Processed data saved to cache.")


def cache_exists(cache_dir):

    cache_path = Path(cache_dir)

    required_files = [
        "text_index.faiss",
        "image_index.faiss",
        "text_metadata.json",
        "image_metadata.json"
    ]

    for filename in required_files:

        if not (cache_path / filename).exists():
            return False

    return True


def load_processed_data(cache_dir):

    cache_path = Path(cache_dir)
    text_index = faiss.read_index(str(cache_path / "text_index.faiss"))
    image_index = faiss.read_index(str(cache_path / "image_index.faiss"))

    with open(
        cache_path / "text_metadata.json",
        encoding="utf-8"
    ) as f:
        text_metadata = json.load(f)

    with open(
        cache_path / "image_metadata.json",
        encoding="utf-8"
    ) as f:
        image_metadata = json.load(f)

    print("Processed data loaded from cache.")

    return (
        text_index,
        image_index,
        text_metadata,
        image_metadata
    )


if __name__ == "__main__":

    print("Loading embeddings...")

    (
        text_vectors,
        image_vectors,
        text_metadata,
        image_metadata
    ) = load_data()

    print(f"Text vectors : {text_vectors.shape}")

    print(f"Image vectors: {image_vectors.shape}")
    print("\nCreating FAISS index...")

    text_index = create_index(text_vectors)
    image_index = create_index(image_vectors)

    faiss.write_index(text_index, str(INDEX_DIR / "text_index.faiss"))

    faiss.write_index(image_index, str(INDEX_DIR / "image_index.faiss"))

    print("\nDone!")

    print("Text index size :", text_index.ntotal)
 
    print("Image index size:", image_index.ntotal) 
