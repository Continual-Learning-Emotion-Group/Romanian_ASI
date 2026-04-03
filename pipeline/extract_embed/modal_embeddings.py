"""
Modal GPU deployment for multilingual embedding inference.

Uses intfloat/multilingual-e5-base (278M params, 768-dim embeddings)
for computing semantic similarity between Romanian ASI sentences.

Usage:
    # Test directly
    modal run pipeline/extract_embed/modal_embeddings.py

    # Called programmatically from run.py
"""

import modal

MODEL_ID = "intfloat/multilingual-e5-base"
VOLUME_NAME = "embedding-model-cache"

app = modal.App("asi-embeddings")

model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "numpy<2",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/cache": model_volume},
    timeout=600,
    scaledown_window=120,
)
class Embedder:
    @modal.enter()
    def load_model(self):
        import os
        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/cache/sentence_transformers"

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_ID, cache_folder="/cache/models")
        # Warm up
        self.model.encode(["test"], normalize_embeddings=True)

    @modal.method()
    def embed_batch(
        self, texts: list[str], prefix: str = "query: ", batch_size: int = 256
    ) -> list[list[float]]:
        """Embed a batch of texts with the given prefix.

        E5 models require prefixes:
        - "query: " for anchors/queries
        - "passage: " for candidate passages
        """
        prefixed = [f"{prefix}{t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()


def embed_texts(
    texts: list[str],
    prefix: str = "query: ",
    batch_size: int = 256,
    chunk_size: int = 2048,
) -> list[list[float]]:
    """Embed texts using Modal GPU inference. Handles app.run() context."""
    from tqdm import tqdm

    all_embeddings = []

    with app.run():
        embedder = Embedder()
        for i in tqdm(range(0, len(texts), chunk_size), desc=f"Embedding ({prefix.strip()})"):
            chunk = texts[i : i + chunk_size]
            embs = embedder.embed_batch.remote(chunk, prefix=prefix, batch_size=batch_size)
            all_embeddings.extend(embs)

    return all_embeddings


@app.local_entrypoint()
def main():
    """Test the embedding function."""
    embedder = Embedder()

    test_sentences = [
        "Mă simt fericit astăzi.",
        "Sunt trist și obosit.",
        "Am mâncat o pizza bună.",
        "Mi-e frică de întuneric.",
    ]

    print(f"Embedding {len(test_sentences)} test sentences...")
    results = embedder.embed_batch.remote(test_sentences, prefix="query: ")
    print(f"Got {len(results)} embeddings, each dim={len(results[0])}")

    import numpy as np
    embs = np.array(results)
    sims = embs @ embs.T
    print("\nCosine similarity matrix:")
    for i, s in enumerate(test_sentences):
        print(f"  [{i}] {s}")
    print(sims.round(3))
