# data_preprocessing.py
import os
import pandas as pd

def read_conll_file(filepath):
    """
    Reads a CoNLL-formatted file and returns a list of sentences.
    Each sentence is represented as a list of (token, tag) tuples.
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as file:
        sentence = []
        for line in file:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                # Assuming the format "word tag"
                parts = line.split()
                if len(parts) >= 2:
                    token, tag = parts[0], parts[-1]
                    sentence.append((token, tag))
        if sentence:
            sentences.append(sentence)
    return sentences

def save_preprocessed_data(sentences, output_file):
    # Save as a simple CSV or pickle (for simplicity in experiments)
    df = pd.DataFrame({"sentence": [" ".join([token for token, tag in s]) for s in sentences],
                       "labels": [ " ".join([tag for token, tag in s]) for s in sentences]})
    df.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    data_path = os.path.join("data", "conll2003_train.txt")
    sentences = read_conll_file(data_path)
    save_preprocessed_data(sentences, os.path.join("data", "conll2003_train.csv"))
    print("Preprocessing complete.")