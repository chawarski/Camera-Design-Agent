import pymupdf
import json
import numpy as np
import ollama
import re
from pathlib import Path

chunk_size = 500
chunk_overlap = 50
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

def split_page(page, page_number: int, image_dir: Path, chapter: str="default"):
    image_dir.mkdir(parents=True, exist_ok=True)
    
    table_regions = []
    table_data = []

    for table_index, table in enumerate(page.find_tables().tables):
        rect = pymupdf.Rect(table.bbox)
        table_regions.append(rect)


        safe_name = re.sub(r'[<>:"/\\|?*]', "", chapter)
        image_path = image_dir / f"{safe_name}-page-{page_number:04d}-table-{table_index:02d}.png"
        page.get_pixmap(clip=rect, dpi=150).save(str(image_path))

        table_data.append({
            "page": page_number,  # 1-based PDF page number
            "bbox": [float(value) for value in table.bbox],
            "image": str(image_path),
        })

    # Group words by their original text block and line, excluding words
    # inside detected table boxes.
    lines: dict[tuple[int, int], list[tuple[int, str]]] = {}

    for word in page.get_text("words"):
        x0, y0, x1, y1, text, block_no, line_no, word_no = word
        word_rect = pymupdf.Rect(x0, y0, x1, y1)

        if any(word_rect.intersects(table_rect) for table_rect in table_regions):
            continue

        lines.setdefault((block_no, line_no), []).append((word_no, text))

    non_table_text = "\n".join(
        " ".join(text for _, text in sorted(lines[key]))
        for key in sorted(lines)
    )

    return non_table_text
        


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
pdf_path = Path(__file__).resolve().parents[3] / "Genie Nano Series User Manual.pdf"
print(pdf_path)
doc = pymupdf.open(str(pdf_path))
# doc = pymupdf.open(r"..\..\..\Genie Nano Series User Manual.pdf")
toc = doc.get_toc()
print(toc)
# print(chapters)

for chapter in chapters:
    if toc[chapter][0] != 2:
        continue
    page_start = toc[chapter][2]-1
    page_end = toc[chapter+1][2]-1
    # raw_text = ",".join(page.get_text() for page in doc if page_start <= page.number <= page_end)
    raw_text = ""
    raw_text = ",".join(split_page(page, page.number, Path("table_images"), toc[chapter][1]) for page in doc if page_start <= page.number <= page_end)
    chunks = chunk_text(raw_text)

    for i, chunk in enumerate(chunks):
        data["slices"].append({
            "chapter": toc[chapter][1],
            "chunk_id": str(i),
            "text": chunk
        })


# for page in doc:
#     if 10 <=page.number <=11:
#         text, tables = split_page(page, page.number, Path("table_images"))
#         print(text)


with open("toc_slicing.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)