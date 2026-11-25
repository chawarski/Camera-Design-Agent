from pathlib import Path
import tomllib
from embedder import TextEmbedder

def main():
    print("Hello, world!")

    # Config
    with Path(r"Camera-Design-Agent\src\CamDesigner\config.toml").open("rb") as file:
        config = tomllib.load(file)

    if True:
        embedder = TextEmbedder(embed_model=config["embed"]["model"], 
                                chunk_size=config["embed"]["chunk_size"], 
                                overlap=config["embed"]["overlap"])
        embedder.make_vectors()

if __name__ == "__main__":
    main()