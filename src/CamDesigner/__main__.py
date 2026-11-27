from pathlib import Path
import tomllib
from .embedder import TextEmbedder
from .slicer import Slicer
from .retriever import TextRetriever
from .table_parser_llm import TableParser

def main():
    print("Starting Cam Designer")

    # Config
    with Path(r"CamDesigner\config.toml").open("rb") as file:
        config = tomllib.load(file)

    if False:
        embedder = TextEmbedder(embed_model=config["embed"]["model"], 
                                chunk_size=config["embed"]["chunk_size"], 
                                overlap=config["embed"]["overlap"])
        embedder.make_vectors()

    if False:
        slicer = Slicer(page_start=0, page_end=335)
        slicer.slice_document()

    if False:
        text_retriever = TextRetriever(embed_model=config["text_retriever"]["embed_model"],
                                       chat_model=config["text_retriever"]["chat_model"],
                                       k=config["text_retriever"]["k"]
                                       )
        question = "Is Virtual Serial Port Driver installed?"
        print("A:" + text_retriever.answer(question))

    if True:
        table_parser = TableParser()
        table_parser.parse_tables()

if __name__ == "__main__":
    main()