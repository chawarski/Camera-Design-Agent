import json
import numpy as np
import ollama
from pathlib import Path

class TextEmbedder:
    def __init__(self, embed_model: str = "nomic-embed-text", chunk_size: int = 100, overlap: int = 10):
        self.embed_model = embed_model
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
                raise ValueError("Require size > overlap >= 0")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        """Takes a long text and split it into chunks with chunk_size and overlap.
        Returns a list of string chunks to be embedded and indexed.
        """
        words = text.split()
        step = self.chunk_size - self.overlap
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), step)]

    def embed(self, text: str) -> np.ndarray:
        """Takes a string and embeds it as a vector. Uses an ollama embedding model.
        Returns np array.
        """
        return np.array(ollama.embed(model=self.embed_model, input=text)["embeddings"][0])

    def make_vectors(self):
        '''Takes a json file populated with text slices from a document.
        Chunks and embedds the slices and then saves them to a local .npz Vector DB.
        '''
        json_path = Path(__file__).resolve().parents[3] / "doc_text_toc_sliced.json"
        with open(json_path, "r", encoding="utf-8") as file:
            slices = json.load(file)["slices"]

        print('..chunking')
        chunks = [chunk for s in slices for chunk in self.chunk_text(s["text"])]
        print('..embedding')
        vectors = np.array([self.embed(chunk) for chunk in chunks])

        print("..saving embedded data")
        npz_path = Path(__file__).resolve().parents[3] / "doc_text_embedded.npz"     
        np.savez_compressed(
                npz_path,
                vectors=vectors,
                chunks=np.asarray(chunks, dtype=np.str_),
            )


