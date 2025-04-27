#!/usr/bin/env python3
import argparse
from tqdm import tqdm

from data_preprocessing import read_conll_file
from bert_ner import BERTNER

def save_predictions(predictions, sentences, output_file):
    import os 
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        # sentences: List of sentences, each sentence = List[(token, gold_tag)]
        for sentence, tags in zip(sentences, predictions):
            for (token, _), pred_tag in zip(sentence, tags):
                f.write(f"{token} {pred_tag}\n")
            f.write("\n")
    print(f"Saved BERT predictions to {output_file}")

def run_bert(input_file: str, output_file: str):
    # 1) read CoNLL data
    test_sentences = read_conll_file(input_file)  # list of [(tok, tag), …]

    # 2) init model
    model = BERTNER()
    results = []

    # 3) predict and rebuild BIO tags
    for sent in tqdm(test_sentences, desc="BERT predictions"): 
        tokens = [tok for tok, _ in sent]
        text   = " ".join(tokens)
        ents   = model.predict(text)

        tags = ["O"] * len(tokens)
        for span in ents:
            label = span["entity_group"]
            start, end = span["start"], span["end"]

            # align char‐offset span back to token indices
            char_idx = 0
            covered = []
            for i, tok in enumerate(tokens):
                tok_str = tok + " "
                tok_start = char_idx
                tok_end   = char_idx + len(tok_str)
                if not (tok_end <= start or tok_start >= end):
                    covered.append(i)
                char_idx += len(tok_str)

            if covered:
                tags[covered[0]] = "B-" + label
                for j in covered[1:]:
                    tags[j] = "I-" + label

        results.append(tags)

    # 4) save
    save_predictions(results, test_sentences, output_file) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run only BERT NER on a CoNLL file")
    parser.add_argument("-i", "--input",  required=True,
                        help="Path to CoNLL-formatted file (e.g. eng.testa)")
    parser.add_argument("-o", "--output", default="results/bert_predictions.txt",
                        help="Where to write token+tag predictions")
    args = parser.parse_args()

    run_bert(args.input, args.output) 