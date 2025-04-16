# main.py
from hmm_model import HMMNER
from crf_model import CRFNER
from bert_ner import BERTNER
from gpt_ner import GPTNER
from transformers import BertTokenizer
import numpy as np
import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm
from data_preprocessing import read_conll_file, save_preprocessed_data

def preprocess_data(train_file, test_file, dev_file=None):
    """
    Preprocess the CoNLL data files.
    
    Args:
        train_file (str): Path to training data
        test_file (str): Path to test data
        dev_file (str, optional): Path to development data
        
    Returns:
        tuple: Preprocessed data for training, testing, and development (if provided)
    """
    print("Preprocessing data...")
    train_sentences = read_conll_file(train_file)
    test_sentences = read_conll_file(test_file)
    dev_sentences = read_conll_file(dev_file) if dev_file else None
    
    # Save preprocessed data
    os.makedirs("processed_data", exist_ok=True)
    save_preprocessed_data(train_sentences, "processed_data/train.csv")
    save_preprocessed_data(test_sentences, "processed_data/test.csv")
    if dev_sentences:
        save_preprocessed_data(dev_sentences, "processed_data/dev.csv")
    
    return train_sentences, test_sentences, dev_sentences

def prepare_hmm_features(sentences):
    """Convert sentences to format suitable for HMM."""
    # Create a vocabulary of words
    vocab = {}
    for sentence in sentences:
        for token, _ in sentence:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    # Convert sentences to sequences of indices
    X = []
    lengths = []
    for sentence in sentences:
        indices = [vocab.get(token, len(vocab)) for token, _ in sentence]  # Use len(vocab) for OOV
        X.extend(indices)
        lengths.append(len(indices))
    
    return np.array(X).reshape(-1, 1), lengths, vocab

def prepare_crf_features(sentences):
    """Convert sentences to format suitable for CRF."""
    X = []
    y = []
    for sentence in sentences:
        X_sent = []
        y_sent = []
        for token, tag in sentence:
            # Create features for each token
            features = {
                "word.lower()": token.lower(),
                "word.isupper()": token.isupper(),
                "word.istitle()": token.istitle(),
                "word.isdigit()": token.isdigit(),
                "word.length": len(token),
            }
            X_sent.append(features)
            y_sent.append(tag)
        X.append(X_sent)
        y.append(y_sent)
    return X, y

def save_predictions(predictions, sentences, output_file):
    """Save model predictions to a file with one word per line and blank lines between sentences."""
    with open(output_file, 'w') as f:
        current_pos = 0
        for sentence in sentences:
            # Get the predictions for this sentence
            sentence_length = len(sentence)
            if isinstance(predictions, list):
                # For CRF and BERT which return list of lists
                if isinstance(predictions[0], list):
                    pred = predictions.pop(0)
                else:
                    # For HMM which returns a flat list
                    pred = predictions[current_pos:current_pos + sentence_length]
                    current_pos += sentence_length
            else:
                # For numpy arrays
                pred = predictions[current_pos:current_pos + sentence_length]
                current_pos += sentence_length
            
            # Write each word and its prediction on a separate line
            for (token, _), tag in zip(sentence, pred):
                f.write(f"{token} {tag}\n")
            # Add blank line between sentences
            f.write("\n")
    print(f"Predictions saved to {output_file}")

def run_hmm(train_sentences, test_sentences):
    print("\nRunning HMM Model:")
    X_train, lengths_train, vocab = prepare_hmm_features(train_sentences)
    X_test, lengths_test, _ = prepare_hmm_features(test_sentences)
    
    model = HMMNER(n_states=9, n_features=len(vocab) + 1)  # +1 for OOV
    print("Training HMM model...")
    model.train(X_train, lengths_train)
    print("Making predictions...")
    pred = model.predict(X_test)
    save_predictions(pred, test_sentences, "results/hmm_predictions.txt")
    print("HMM predictions shape:", pred.shape)

def run_crf(train_sentences, test_sentences):
    print("\nRunning CRF Model:")
    X_train, y_train = prepare_crf_features(train_sentences)
    X_test, y_test = prepare_crf_features(test_sentences)
    
    model = CRFNER()
    print("Training CRF model...")
    model.train(X_train, y_train)
    print("Making predictions...")
    pred = model.predict(X_test)
    save_predictions(pred, test_sentences, "results/crf_predictions.txt")
    print("CRF predictions for first sentence:", pred[0])

def run_bert(test_sentences):
    print("\nRunning BERT-based NER:")
    model = BERTNER()
    results = []
    print("Processing sentences...")
    for sentence in tqdm(test_sentences, desc="BERT predictions"):
        sample_text = " ".join([token for token, _ in sentence])
        result = model.predict(sample_text)
        # Convert BERT output to list of tags
        tags = []
        for entity in result:
            if isinstance(entity, dict):
                tags.append(entity.get('label', 'O'))
            else:
                tags.append('O')
        results.append(tags)
    
    save_predictions(results, test_sentences, "results/bert_predictions.txt")
    print("BERT predictions saved to results/bert_predictions.txt")

def run_gpt(test_sentences):
    print("\nRunning GPT-based NER:")
    model = GPTNER()
    results = []
    print("Processing sentences...")
    for sentence in tqdm(test_sentences, desc="GPT predictions"):
        sample_text = " ".join([token for token, _ in sentence])
        prompt = f"Extract the named entities from the following sentence: '{sample_text}' Entities:"
        result = model.predict(prompt)
        # Convert GPT output to list of tags (this is a simplified version)
        # You might need to adjust this based on your specific GPT output format
        tags = ['O'] * len(sentence)  # Default to 'O' for all tokens
        results.append(tags)
    
    save_predictions(results, test_sentences, "results/gpt_predictions.txt")
    print("GPT predictions saved to results/gpt_predictions.txt")

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Run NER models with CoNLL data')
    parser.add_argument('--train', default='eng.train', help='Path to training data')
    parser.add_argument('--test', default='eng.testb', help='Path to test data')
    parser.add_argument('--dev', default='eng.testa', help='Path to development data')
    parser.add_argument('--models', nargs='+', default=['hmm', 'crf', 'bert', 'gpt'],
                        help='Models to run (hmm, crf, bert, gpt)')
    args = parser.parse_args()
    
    # Preprocess data
    print("Preprocessing data...")
    train_sentences, test_sentences, dev_sentences = preprocess_data(args.train, args.test, args.dev)
    print(f"Loaded {len(train_sentences)} training sentences, {len(test_sentences)} test sentences")
    
    # Run selected models
    model_funcs = {
        'hmm': lambda: run_hmm(train_sentences, test_sentences),
        'crf': lambda: run_crf(train_sentences, test_sentences),
        'bert': lambda: run_bert(test_sentences),
        'gpt': lambda: run_gpt(test_sentences)
    }
    
    for model in args.models:
        if model.lower() in model_funcs:
            try:
                print(f"\nStarting {model.upper()} model...")
                model_funcs[model.lower()]()
            except Exception as e:
                print(f"Error running {model} model: {str(e)}")
        else:
            print(f"Unknown model: {model}")

if __name__ == "__main__":
    main()
