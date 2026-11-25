import json
import tomllib
import numpy as np
import ollama
from pathlib import Path

# Config
# with Path(r"Camera-Design-Agent\src\CamDesigner\config.toml").open("rb") as file:
#     config = tomllib.load(file)
# EMBED_MODEL = config["embed"]["model"]
# CHUNK_SIZE = config["embed"]["chunk_size"]
# CHUNK_OVERLAP = 10

class TextEmbedder:
    def __init__(self, embed_model: str = "nomic-embed-text", chunk_size: int = 100, overlap: int = 10):
        self.embed_model = embed_model
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
                raise ValueError("Require size > overlap >= 0")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        words = text.split()
        step = self.chunk_size - self.overlap
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), step)]

    def embed(self, text: str) -> np.ndarray:
        print('embedding')
        return np.array(ollama.embed(model=self.embed_model, input=text)["embeddings"][0])

    def make_vectors(self):
        with open("doc_text_toc_sliced.json", "r", encoding="utf-8") as file:
            slices = json.load(file)["slices"]

        # print(slices[0])
        # chunks = []
        # texts = [s["text"] for s in slices]
        # for text in texts:
        #         for chunk in self.chunk_text(text):
        #             chunks.append(chunk) 
        chunks = [chunk for s in slices for chunk in self.chunk_text(s["text"])]
        vectors = np.array([self.embed(chunk) for chunk in chunks])

        print("..saving embedded data")
        np.savez_compressed(
                f"doc_text_embedded.npz",
                vectors=vectors,
                chunks=np.asarray(chunks, dtype=np.str_),
            )


