import pymupdf
import json
import numpy as np
import ollama
import re
from pathlib import Path

chunk_size = 500
chunk_overlap = 50
CHAT_MODEL = "llama3.2"

class Slicer:
    def __init__(self, page_start: int = 0, page_end: int = 1):
        self.page_start = page_start
        self.page_end = page_end
            
# # Chunk and Index the text
# def chunk_text(text: str, size: int = chunk_size, overlap: int = chunk_overlap) -> list[str]:
#     # print('chunking')
#     if size <= 0 or overlap < 0 or overlap >= size:
#         raise ValueError("Require size > overlap >= 0")

#     words = text.split()
#     step = size - overlap
#     return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]

# def route(question: str) -> str:
#     print('answering')
#     # print(f"\nQ: {question}\nRetrieved:")
#     # print(context)
#     prompt = (
#          f"Write for LLM ingestion, minimum length, no preamble just output. Write a short technical paragraph of features described in this Context:\n{question}"
#     )
#     reply = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
#     return reply["message"]["content"]

    def split_page(self, page, page_number: int, chapter: str="default"):
        ''' Splits a pdf document into table areas and non-table areas.
            Iterates through document by chapter.
            Saves table images, each table as a png file into table_images directories
            Saves non-table text from each chapter to file
        '''

        image_dir = Path(__file__).resolve().parents[3] / "table_images" 
        image_dir.mkdir(parents=True, exist_ok=True)
        text_dir = Path(__file__).resolve().parents[3] / "chapter_text" 
        text_dir.mkdir(parents=True, exist_ok=True)
        
        table_regions = []
        table_data = []
        name = re.sub(r'[<>:"/\\|?*]', "", chapter)

        for table_index, table in enumerate(page.find_tables().tables):
            if page.number < self.page_start or page.number > self.page_end:
                # skip if pages are outside requested range
                continue

            rect = pymupdf.Rect(table.bbox)
            table_regions.append(rect)

            image_path = image_dir / f"{name}-page-{page_number:04d}-table-{table_index:02d}.png"
            page.get_pixmap(clip=rect, dpi=150).save(str(image_path))

            # table_data.append({
            #     "page": page_number,  # 1-based PDF page number
            #     "bbox": [float(value) for value in table.bbox],
            #     "image": str(image_path),
            # })

        # Group words by their original text block and line, excluding words inside detected table boxes.
        lines: dict[tuple[int, int], list[tuple[int, str]]] = {}

        for word in page.get_text("words"):
            x0, y0, x1, y1, text, block_no, line_no, word_no = word
            word_rect = pymupdf.Rect(x0, y0, x1, y1)

            # Skip any words inside table rect areas
            if any(word_rect.intersects(table_rect) for table_rect in table_regions):
                continue

            lines.setdefault((block_no, line_no), []).append((word_no, text))

        non_table_text = "\n".join(
            " ".join(text for _, text in sorted(lines[key]))
            for key in sorted(lines)
        )
        
        # text_path = text_dir / f"{name}-page-{page_number:04d}-text.txt"
        text_path = Path(__file__).resolve().parents[3] / text_dir / f"{name}-page-{page_number:04d}-text.txt"
        text_path.write_text(non_table_text, encoding="utf-8")
        return non_table_text
            
    def slice_document(self):
        data = {
            "document": {
            "title": "Genie Nano Series Camera User's Manual",
            "total_pages": 335
        },
        "slices": []
        }
        
        pdf_path = Path(__file__).resolve().parents[3] / "Genie Nano Series User Manual.pdf"
        doc = pymupdf.open(str(pdf_path))
        # toc = Table of Contents. [0] is heirarchy [1] is name [2] is start page
        toc = doc.get_toc()
        chapters = tuple(range(len(toc)-1))


        for chapter in chapters:
            if toc[chapter][0] != 2:
                # Use only sub-chapters and skip some chapter heirarchy to avoid repeat sections
                continue
            page_start = toc[chapter][2]-1 # start at start page index
            page_end = toc[chapter+1][2]-1 # end at the next chapter start page index
            print(f"..slicing {chapter}: {toc[chapter][1]}")
            raw_text = ""
            raw_text = ",".join(self.split_page(page, page.number, toc[chapter][1]) for page in doc if page_start <= page.number <= page_end)

            data["slices"].append({
                                "chapter": toc[chapter][1],
                                "text": raw_text
                            })

        json_path = Path(__file__).resolve().parents[3] / "doc_text_toc_sliced.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print("Saved doc_text_toc_sliced.json")



