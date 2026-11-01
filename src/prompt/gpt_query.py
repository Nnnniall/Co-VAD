import os
import json
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

API_SECRET_KEY = "YOUR_API_KEY"
BASE_URL = "https://flag.smarttrot.com/v1/"

os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=250
)

parser = StrOutputParser()

# UCF-Crime
class_names = [
    "normal", "abuse", "arrest", "arson", "assault", "burglary", "explosion",
    "fighting", "road accidents", "robbery", "shooting", "shoplifting", "stealing", "vandalism"
]

# XD-Violence
# class_names = [
#     "normal", "fighting", "shooting", "riot",
#     "abuse", "car accident", "explosion"
# ]

NORMAL_TEMPLATES = [
    "describe a surveillance scene with routine pedestrian and vehicle movement.",
    "describe ordinary object usage and human interaction patterns indicating daily activity.",
    "describe a surveillance scene where behavior clearly follows normal daily activity.",
    "describe concrete motion patterns and interactions that commonly occur in ordinary situations.",
    "write a realistic surveillance caption highlighting routine interactions and objects.",
    "describe role-symmetric interactions and environment-compliant behavior characteristic of daily scenes."
]

# ============== ABNORMAL PROMPT TEMPLATES ==============
ABNORMAL_TEMPLATES = [
    "describe distinctive visual interactions typically observed in {cls}.",
    "describe unusual object usage and human interaction patterns indicating {cls}.",
    "describe a surveillance scene where behavior clearly deviates from normal daily activity in {cls}.",
    "describe concrete motion patterns and interactions that would rarely occur in ordinary situations related to {cls}.",
    "write a realistic surveillance caption highlighting discriminative interactions and objects in {cls}.",
    "describe role-asymmetric interactions and environment violations characteristic of {cls}."
]

NUM_SAMPLES_PER_TEMPLATE = 5
TARGET_PER_CLASS = 30
OUTPUT_JSON = "ucf_prompt.json"

def clean_sentences(text):
    lines = []
    for line in text.split("\n"):
        line = line.strip().lstrip("0123456789.-• ")
        if 6 <= len(line.split()) <= 18:
            lines.append(line.lower())
    return lines

all_responses = {}
for cls in tqdm(class_names):
    collected = []
    templates = NORMAL_TEMPLATES if cls == "normal" else ABNORMAL_TEMPLATES
    
    for template in templates:
        if len(collected) >= TARGET_PER_CLASS:
            break

        instruction = (
            "you are an expert in video anomaly detection.\n"
            f"generate {NUM_SAMPLES_PER_TEMPLATE} distinct sentences.\n"
            "each sentence must be 10–15 words.\n"
            "return each sentence on a new line without numbering.\n"
            "describe only visible people, objects, vehicles, and interactions.\n"
            "avoid abstract words.\n"
            "do not explicitly mention the event category name.\n"
            "focus on distinctive interactions, object usage, or role asymmetry.\n"
            "the description must clearly differ from normal walking, standing, or traffic flow.\n"
            "avoid generic aggressive motion such as running or crowd movement without context.\n"
            "each sentence should help a model distinguish this from normal daily behavior.\n"
            "use lowercase.\n\n"
            + template
        )

        prompt = ChatPromptTemplate.from_template(instruction)
        chain = prompt | model | parser

        response = chain.invoke({"cls": cls})
        sentences = clean_sentences(response)

        for s in sentences:
            if len(collected) < TARGET_PER_CLASS:
                collected.append(s)

    collected = list(dict.fromkeys(collected))
    all_responses[cls] = collected[:TARGET_PER_CLASS]

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_responses, f, indent=4, ensure_ascii=False)

print(f"Saved {TARGET_PER_CLASS} prompts per class to ./{OUTPUT_JSON}")
