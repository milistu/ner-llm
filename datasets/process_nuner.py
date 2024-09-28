import ast
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from datasets import Dataset, load_dataset


def tokenize_text(text):
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r"\w+(?:[-_]\w+)*|\S", text)


def process_entities(dataset):
    """Processes entities in the dataset to extract tokenized text and named entity spans."""
    all_data = []
    for el in tqdm(dataset["entity"]):
        try:
            tokenized_text = tokenize_text(el["input"])
            parsed_output = ast.literal_eval(el["output"])
            entity_texts, entity_types = zip(*[i.split(" <> ") for i in parsed_output])

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

        except Exception as e:
            print(f"Error {e} processing entry.")
            continue

        all_data.append({"tokenized_text": tokenized_text, "ner": entity_spans})
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
    for id, entry in enumerate(processed_data):
        try:
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

            labels = ["O"]
            for label in temp_labels:
                label = label.upper().replace(" ", "_")
                labels.extend([f"B-{label}", f"I-{label}"])

            rows.append({"id": id, "words": tokens, "ner_tags": tags, "labels": labels})
        except Exception as e:
            print(f"Error {e} processing entry: {id}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    dataset_id = "numind/NuNER"
    cache_dir = Path("./cache")

    output_path = Path("./datasets/processed_data/nuner")
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_id, cache_dir=cache_dir)
    processed_data = process_entities(dataset)
    print("dataset size:", len(processed_data))

    save_data_to_file(processed_data, output_path / "nuner-train-gliner.jsonl")

    df = convert_to_conll_rows(processed_data)
    df.to_json(output_path / "nuner-train.jsonl", orient="records", lines=True)

    # Push to HF Datasets
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(repo_id="Studeni/NuNER-conll", split="train")
