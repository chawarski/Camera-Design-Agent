import numpy as np
import ollama
import json

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

def embed(text: str) -> np.ndarray:
    print('embedding')
    return np.array(ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0])

# 3. RETRIEVE -- cosine similarity between the question and each document
def retrieve(question: str, k: int = 2) -> list[str]:
    with np.load(f"toc_sliced_embedded.npz", allow_pickle=False) as index:
        vectors = index["vectors"]
        chunks = index["chunks"]
    print('retrieving')
    q = embed(question)
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(scores)[::-1][:k]
    # for i in top:
    #   print(f" [{scores[i]:.3f}] {chunks[i][:70]}...")
    return [chunks[i] for i in top]

# # 4. AUGMENT + GENERATE -- stuff retrieved context into the prompt
def answer(question: str) -> str:
    print('answering')
    # print(f"\nQ: {question}\nRetrieved:")
    context = "\n".join(f"- {c}" for c in retrieve(question))
    # context = slices
    # print(context)
    prompt = (
        "Answer using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    reply = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return reply["message"]["content"]

with open(r"eval\camera.json", "r", encoding="utf-8") as f:
    slices = json.load(f)

q = "How to test Nano with Lens"
print("A:" + answer(q))