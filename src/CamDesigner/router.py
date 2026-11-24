import ollama
from pydantic import BaseModel, Field
import re
from typing import Literal

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3.5"

class RouteDecision(BaseModel):
    route: Literal["table_lookup", "text_chunks", "reject"] = Field(
        description=(
            "Strictly choose ONE: "
            "'table_lookup' for exact numerical specs, part numbers, pinouts, resolutions, frame rates, firmware filenames, or comparing models; "
            "'text_chunks' for setup guides, troubleshooting, wiring instructions, conceptual explanations, or feature behavior;"
            "'reject' for nonsensical text, gibberish, general chit-chat, or questions unrelated to cameras, machine vision, or imaging systems."
        )
    )


def route(question: str) -> RouteDecision:
    print('thinking..')
    ROUTER_SYSTEM_PROMPT = """You are a query router for a technical camera documentation system.
        Classify the user's question into ONE of: "table_lookup" OR "text_chunks" OR "reject" and return valid JSON.

        <routing_rules>
        1. "reject": use FIRST for nonsensical text, gibberish, general chit-chat, or questions unrelated to cameras, machine vision, or imaging systems."
        2. "table_lookup": Use when the answer depends on structured specification data—exact numbers, resolutions, pixel sizes, frame rates, part numbers, firmware filenames, connector pinouts, or comparing multiple camera models.
        3. "text_chunks": Use when the answer depends on prose explanations—how-to configuration steps, troubleshooting, theory of operation, timing behavior, firmware, or environmental/handling guidelines.
        </routing_rules>

        <examples>
        <example>
        <question>What is the pixel size and max frame rate of the Genie Nano M1940?</question>
        <output>{"route": "table_lookup"}</output>
        </example>

        <example>
        <question>Which 5MP color camera has the highest dynamic range?</question>
        <output>{"route": "table_lookup"}</output>
        </example>

        <example>
        <question>How do I prevent packet drops when using TurboDrive with multiple cameras?</question>
        <output>{"route": "text_chunks"}</output>
        </example>

        <example>
        <question>Why is my C4900 showing dark vertical columns in bright areas?</question>
        <output>{"route": "text_chunks"}</output>
        </example>

        <example>
        <question>What is the best recipe for sourdough bread?</question>
        <output>{"route": "reject"}</output>
        </example>

        <example>
        <question>asdf 1234 banana purple flying toaster</question>
        <output>{"route": "reject"}</output>
        </example>
        </examples>"""
    prompt = (
        "You are routing a query for a technical manual. \n"
        "Depending on the details of the question, the answer context will come from text chunks or from table data. \n"
        "You need to choose from two options: \n" 
        " 1. TABLE DATA: containing specifications, dimensions, resolutions, connections, standards, compliances \n"
        " 2. TEXT CHUNKS: containing marketing info, use case explanations, instructions, feature overview, manufacturer recommendations.\n"
        " Choose 1 or 2 based on the question and write 1 or 2 in the CONTEXT Field in the format.\n"
        " Write a one sentence summary of the decision and put it in the JUSTIFICATION Field in the format. \n"
    )
    reply = ollama.chat(
        model=CHAT_MODEL, 
        messages=[
        {
            "role": "system",
            "content": ROUTER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"<question>{question}</question>",
        },
        ],
        format=RouteDecision.model_json_schema(),
        options={"num_gpu": 99, "num_ctx": 6144, "num_predict": 4096, "temperature": 0.0})
    raw_content = reply.message.content

    clean_json = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

    parsed = RouteDecision.model_validate_json(clean_json)
    # print(parsed.model_dump_json(indent=2))
    
    return parsed

q = "What is the sensor model in M640 model?"
parsed = route(q)
print(parsed)