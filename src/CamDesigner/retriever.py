import numpy as np
import ollama
import json
from pydantic import BaseModel
import re

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "phi4-mini-reasoning"


class ExtractedSpec(BaseModel):
    context: str
    justification: str



def embed(text: str) -> np.ndarray:
    print('embedding')
    return np.array(ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0])

# RETRIEVE -- cosine similarity between the question and each document
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

def answer(question: str) -> str:
    print('answering')
    # print(f"\nQ: {question}\nRetrieved:")
    # context = "\n".join(f"- {c}" for c in retrieve(question))
    context = slices
    # print(context)
    prompt = (
        "Answer using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    token_estimate = len(str(prompt))/4
    print("tokens: " + str(token_estimate))
    reply = ollama.chat(
        model=CHAT_MODEL, 
        messages=[{"role": "user", "content": prompt}],
        options={"num_gpu": 99, "num_ctx": 6144, "num_predict": 4096, "temperature": 0.0})
    return reply["message"]["content"]



with open(r"table_data.json", "r", encoding="utf-8") as f:
    slices = json.load(f)

def decide(question: str) -> str:
    print('answering')
    # print(f"\nQ: {question}\nRetrieved:")
    # context = "\n".join(f"- {c}" for c in retrieve(question))
    context = slices
    # print(context)
    prompt = (
        "You are routing a query for a technical manual. \n"
        "Depending on the details of the question, the answer context will come from text chunks or from table data. \n"
        "You need to choose from two options: \n" 
        " 1. TABLE DATA: containing specifications, dimensions, resolutions, connections, standards, compliances \n"
        " 2. TEXT CHUNKS: containing marketing info, use case explanations, instructions.\n"
        " Choose 1 or 2 based on the question and write 1 or 2 in the CONTEXT Field in the format.\n"
        " Write a one sentace summary of the decision and put it in the JUSTIFICATION Field in the format. \n"
    )
    reply = ollama.chat(
        model=CHAT_MODEL, 
        messages=[
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": question,
        },
        ],
        format=ExtractedSpec.model_json_schema(),
        options={"num_gpu": 99, "num_ctx": 6144, "num_predict": 4096, "temperature": 0.0})
    raw_content = reply.message.content

    clean_json = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

    parsed = ExtractedSpec.model_validate_json(clean_json)
    # print(parsed.model_dump_json(indent=2))
    
    return parsed.model_dump_json(indent=2)

q = "What is the sensor type in model M640?"
print("A:" + decide(q))