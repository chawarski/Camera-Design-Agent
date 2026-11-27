import numpy as np
import ollama
# from pydantic import BaseModel
from pathlib import Path

# class ExtractedSpec(BaseModel):
#     context: str
#     justification: str

class TextRetriever:
    def __init__(self, embed_model: str = "nomic-embed-text", chat_model: str = "llama3.2", k: int = 3):
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.k = k

     
    def embed(self, text: str) -> np.ndarray:
        '''Embeds the question text and returns vector for context matching.'''
        print('..embedding question')
        return np.array(ollama.embed(self.embed_model, input=text)["embeddings"][0])


    def retrieve(self, question: str, vectors, chunks) -> list[str]:
        '''Runs a similarity algorithm between the quesiton and context vectors.
        Returns the top matching context chunks for use by chat prompt.'''
        print('..retrieving')
        q = self.embed(question)
        scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
        top = np.argsort(scores)[::-1][:self.k]
        return [chunks[i] for i in top if scores[i] > 0.35]


    def answer(self, question: str) -> str:
        '''RAG. Retrieves context based on question and propmts LLM for answer.
        Returns LLM answer string.'''
        print('..answering')
        npz_dir = Path(__file__).resolve().parents[3] / "doc_text_embedded.npz" 
        with np.load(npz_dir, allow_pickle=False) as index:
                vectors = index["vectors"]
                chunks = index["chunks"]
        context = "\n".join(f"- {c}" for c in self.retrieve(question, vectors, chunks))
        print(context)
        prompt = (
            "Answer using ONLY the context below. Summarize answer to one sentence."
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        token_estimate = len(str(prompt))/4
        print("estimated tokens: " + str(token_estimate))
        reply = ollama.chat(
            model=self.chat_model, 
            messages=[{"role": "user", "content": prompt}],
            options={"num_gpu": 99, "num_ctx": 6144, "num_predict": 4096, "temperature": 0.0})
        return reply["message"]["content"]