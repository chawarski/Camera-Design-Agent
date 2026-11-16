import pymupdf
import json
import numpy as np
import ollama

chunk_size = 1000
chunk_overlap = 100
CHAT_MODEL = "llama3.2"

# Chunk and Index the text
def chunk_text(text: str, size: int = chunk_size, overlap: int = chunk_overlap) -> list[str]:
    # print('chunking')
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")

    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]

def route(question: str) -> str:
    print('answering')
    # print(f"\nQ: {question}\nRetrieved:")
    # print(context)
    prompt = (
         f"Write for LLM ingestion, minimum length, no preamble just output. Write a short technical paragraph of features described in this Context:\n{question}"
    )
    reply = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return reply["message"]["content"]

data = {
    "document": {
    "title": "Genie Nano Series Camera User's Manual",
    "part_number": "G3-G00M-USR00",
    "revision": "0025",
    "total_pages": 335
  },
  "slices": []
}

chapters = tuple(range(335))
doc = pymupdf.open(r"eval\Genie Nano Series User Manual.pdf")
toc = doc.get_toc()
print(toc)
print(chapters)

for chapter in chapters:
    page_start = toc[chapter][2]-1
    page_end = toc[chapter+1][2]-1
    raw_text = ",".join(page.get_text() for page in doc if page_start <= page.number <= page_end)
    chunks = chunk_text(raw_text)

    for i, chunk in enumerate(chunks):
        data["slices"].append({
            "chapter": toc[chapter][1],
            "chunk_id": str(i),
            "text": chunk
        })




with open("toc_slicing.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)