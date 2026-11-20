from google import genai
import ollama
import os
import base64
from pathlib import Path
import json
import shutil
from pydantic import BaseModel, Field

use_gemini = False



# Pattern A: Dynamic schema for ANY unknown table layout (keeps header + cell grouped per row)
class CellPair(BaseModel):
    header: str = Field(description="Exact column header name for this cell")
    value: str = Field(description="Text content inside this cell")

class TableRow(BaseModel):
    row_index: int
    cells: list[CellPair]

class ExtractedTable(BaseModel):
    table_title: str
    headers: list[str]
    rows: list[TableRow]

class CompactTable(BaseModel):
    table_title: str
    headers: list[str]
    rows: list[list[str]]  # Each inner list aligns positionally with headers

# Create a directory for processed images. Images will be moved here after they are converted to JSON
destination_folder = Path("processed_tables")
destination_folder.mkdir(parents=True, exist_ok=True)

path = Path(r"table_data.json")

if path.exists():
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
else:
    print("It does not exist")
    data = {
            "document": {
              "title": "Genie Nano Series Camera User's Manual"
            },
            "tables": []
        }


if use_gemini:
    # Set model to use for the chatbot, from https://aistudio.google.com/docs/models
    # model = "gemini-3.8-flash"
    model = "gemini-4-hydrogen"
    tokens_used = 0 # Initialize token usage counter
    api_key = os.environ["MY_API_KEY"]
    try:
        # Create a client instance with your API key
        print("..Initializing gemini")
        client = genai.Client(api_key=api_key)
        
        # Send a tiny prompt to verify end-to-end execution
        response = client.interactions.create(
            model=model,
            input="ping",
        )
            
    except Exception as e:
        raise SystemExit(f"Exiting due to failed Gemini AI client connection: {e}")

    # Welcome message indicating good connection to Gemini AI and the model being used
    print(f"Hello, this is Gemini AI chatbot, you are speaking to {model}")


for image_path in sorted(Path("table_images").glob("*.png")):
    print(f"...processing {image_path.name}")
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

# image_path = Path(r"table_images\Overview-page-0257-table-00.png")
# image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    schema = CompactTable.model_json_schema()
    # focused_query = f"Use image data of a table and parse it. Convert table data to llm readable JSON format. Keep rows and headings together. Give response in JSON format."
    focused_query = f"""Extract the table from this image into row-by-row JSON.
        Rules:
        1. Do NOT output bounding boxes, coordinates, or pixel locations.
        2. Group cells horizontally by row, pairing each cell value with its exact column header.
        3. If a row has merged/split sub-rows (such as C-mount and CS-mount), repeat the parent row values so every row is complete.
        4. Return ONLY valid JSON matching this schema:
        {schema}"""
    if not use_gemini:
        print("..running query with ollama")

        

        response =ollama.chat(
            model="qwen2.5vl",
            messages=[
                {
                    "role": "user",
                    "content": (focused_query),
                    "images": [image_path],
                }
            ],
            format=schema,
            options={"num_gpu": 99, "num_ctx": 6144, "num_predict": 4096, "temperature": 0.0},
            )
        print(response["message"]["content"])
        if response.get("done_reason") == "length":
            raise RuntimeError(f"Output truncated at {response.get('eval_count')} tokens—split the table image horizontally.")

        table_data = CompactTable.model_validate_json(response["message"]["content"])
        # print("message:" + str(response["message"]))
        # print("content:" + str(response["content"]))

        # table_json = json.loads(response["message"])
        data["tables"].append({
            "name": image_path.name,
            "data": table_data.model_dump(mode="json")
        })


    if use_gemini:
        interaction = client.interactions.create(
            model=model,
            input=[
                {"type": "text", "text": focused_query},
                {
                    "type": "image",
                    "data": image_data,
                    "mime_type": "image/png",
                },
            ],
        )
        print(f"  {interaction.output_text}")
        response_text = interaction.output_text
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0].strip()
            table_json = json.loads(response_text)
            data["tables"].append({
                "name": image_path.name,
                "data": table_json
            })

    print(response["message"]["content"])

    # response_text = interaction.output_text
    # if response_text.startswith("```"):
    #     response_text = response_text.split("\n", 1)[1]
    #     response_text = response_text.rsplit("```", 1)[0].strip()
    #     table_json = json.loads(response_text)
    #     data["tables"].append({
    #         "name": image_path.name,
    #         "data": table_json
    #     })

with open("table_data.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

    #     shutil.move(str(image_path), str(destination_folder / image_path.name))
    # else:
    #     print(f"format issue with llm response at {image_path.name}")
    #     break




# Loop script
# while True:
#     query = input("Enter your query: ") # User query request in terminal
#     if query.lower() == "exit": # Exit condition to break the loop
#         print("Exiting the chatbot. Goodbye!")
#         break
#     focused_query = f"answer in one sentence: {query}" # Keep chatbot answer concise and focused to one sentence
#     # API call to generate response
#     interaction = client.interactions.create(
#         model=model,
#         input=focused_query
#     )
#     # tokens_used += interaction.usage.total_tokens # Update token usage counter

#     print(f"  {interaction.output_text}") # Print the model response in terminal
#     # print(f"  Total tokens used: {tokens_used}") # Print the total tokens used so far in terminal