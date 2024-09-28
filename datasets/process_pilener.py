import ast
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from datasets import Dataset, load_dataset


def load_data(filepath):
    """Loads data from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def load_jsonl(filepath):
    """Loads data from a JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def tokenize_text(text):
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r"\w+(?:[-_]\w+)*|\S", text)


def extract_entity_spans(entry):
    """Extracts entity spans from an entry."""
    id = entry["id"]

    len_start = len("What describes ")
    len_end = len(" in the text?")
    entity_types, entity_texts, negative = [], [], []

    for c in entry["conversations"]:
        if c["from"] == "human" and c["value"].startswith("Text: "):
            text = c["value"][len("Text: ") :]
            tokenized_text = tokenize_text(text)
        elif c["from"] == "human" and c["value"].startswith("What describes "):
            entity_type = c["value"][len_start:-len_end]
            entity_types.append(entity_type)
        elif c["from"] == "gpt" and c["value"].startswith("["):
            if c["value"] == "[]":
                negative.append(entity_types.pop())
                continue
            texts_ents = ast.literal_eval(c["value"])
            entity_texts.extend(texts_ents)
            num_repeat = len(texts_ents) - 1
            entity_types.extend([entity_types[-1]] * num_repeat)

    entity_spans = []
    for j, entity_text in enumerate(entity_texts):
        entity_tokens = tokenize_text(entity_text)
        matches = []
        for i in range(len(tokenized_text) - len(entity_tokens) + 1):
            if (
                " ".join(tokenized_text[i : i + len(entity_tokens)]).lower()
                == " ".join(entity_tokens).lower()
            ):
                matches.append((i, i + len(entity_tokens) - 1, entity_types[j]))
        if matches:
            entity_spans.extend(matches)

    return {
        "id": id,
        "tokenized_text": tokenized_text,
        "ner": entity_spans,
        "negative": negative,
    }


def process_data(data):
    """Processes a list of data entries to extract entity spans."""
    all_data = [extract_entity_spans(entry) for entry in tqdm(data)]
    return all_data


def save_data_to_file(data, filepath):
    """Saves the processed data to a JSONL file."""
    with open(filepath, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def convert_to_conll_rows(processed_data: List[Dict]) -> pd.DataFrame:
    """
    Convert processed NER data to a CONLL-like format.

    Args:
    processed_data (List[Dict]): List of dictionaries containing 'tokenized_text' and 'ner' keys.

    Returns:
    pd.DataFrame: DataFrame with columns 'id', 'word', and 'ner_tag'.
    """
    rows = []
    for entry in processed_data:
        try:
            id = entry["id"]
            tokens = entry["tokenized_text"]
            ner_spans = entry["ner"]

            # Initialize all tags to 'O'
            tags = ["O"] * len(tokens)

            temp_labels = set()
            # Fill in the B- and I- tags
            for start, end, entity_type in ner_spans:
                temp_labels.add(entity_type)
                if start > end:
                    print(f"Invalid span: {start}-{end}, ID: {id}")
                    continue

                entity_type = entity_type.upper().replace(" ", "_")
                tags[start] = f"B-{entity_type}"
                for i in range(start + 1, end + 1):
                    tags[i] = f"I-{entity_type}"

            temp_labels.update(entry["negative"])

            labels = ["O"]
            for label in temp_labels:
                label = label.upper().replace(" ", "_")
                labels.extend([f"B-{label}", f"I-{label}"])

            rows.append({"id": id, "words": tokens, "ner_tags": tags, "labels": labels})
        except Exception as e:
            print(f"Error {e} processing entry: {id}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    dataset_id = "Universal-NER/Pile-NER-type"
    cache_dir = Path("./cache")

    dataset = load_dataset(path=dataset_id, cache_dir=cache_dir, split="train")
    print(f"Number of data samples in the dataset: {len(dataset)}")

    processed_data = process_data(dataset)
    df = convert_to_conll_rows(processed_data)

    output_path = Path("./datasets/processed_data/pilener")
    output_path.mkdir(parents=True, exist_ok=True)

    save_data_to_file(processed_data, output_path / "pilener-train-gliner.jsonl")
    df.to_json(output_path / "pilener-train.jsonl", orient="records", lines=True)

    # Push to HF Datasets
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(repo_id="Studeni/Pile-NER-type-conll", split="train")
