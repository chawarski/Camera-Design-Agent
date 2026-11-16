import json

import numpy as np
import ollama
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

def embed(text: str) -> np.ndarray:
    print('embedding')
    return np.array(ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0])

with open("toc_slicing.json", "r", encoding="utf-8") as f:
    slices = json.load(f)["slices"]

texts = [s["text"] for s in slices]
vectors = np.array([embed(d) for d in texts])

print("saving embedded data")
np.savez_compressed(
        f"toc_sliced_embedded.npz",
        vectors=vectors,
        chunks=np.asarray(texts, dtype=np.str_),
    )


