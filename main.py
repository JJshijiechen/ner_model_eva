# main.py
from hmm_model import HMMNER
from crf_model import CRFNER
from bert_ner import BERTNER
from gpt_ner import GPTNER
from hybrid_model import BERT_CRF
from transformers import BertTokenizer
import numpy as np
import torch
import argparse
import os
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

def run_hmm(train_sentences, test_sentences):
    print("\nRunning HMM Model:")
    X_train, lengths_train, vocab = prepare_hmm_features(train_sentences)
    X_test, lengths_test, _ = prepare_hmm_features(test_sentences)
    
    model = HMMNER(n_states=9, n_features=len(vocab) + 1)  # +1 for OOV
    model.train(X_train, lengths_train)
    pred = model.predict(X_test)
    print("HMM predictions shape:", pred.shape)

def run_crf(train_sentences, test_sentences):
    print("\nRunning CRF Model:")
    X_train, y_train = prepare_crf_features(train_sentences)
    X_test, y_test = prepare_crf_features(test_sentences)
    
    model = CRFNER()
    model.train(X_train, y_train)
    pred = model.predict(X_test)
    print("CRF predictions for first sentence:", pred[0])

def run_bert(test_sentences):
    print("\nRunning BERT-based NER:")
    model = BERTNER()
    # Take first test sentence as example
    sample_text = " ".join([token for token, _ in test_sentences[0]])
    result = model.predict(sample_text)
    print("BERT NER output:")
    for entity in result:
        print(entity)

def run_gpt(test_sentences):
    print("\nRunning GPT-based NER (via prompt engineering):")
    model = GPTNER()
    # Take first test sentence as example
    sample_text = " ".join([token for token, _ in test_sentences[0]])
    prompt = f"Extract the named entities from the following sentence: '{sample_text}' Entities:"
    result = model.predict(prompt)
    print("GPT NER output:")
    print(result)

def run_hybrid(test_sentences):
    print("\nRunning Hybrid BERT+CRF Model:")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # Take first test sentence as example
    sample_text = " ".join([token for token, _ in test_sentences[0]])
    inputs = tokenizer(sample_text, return_tensors="pt")
    model = BERT_CRF(bert_model_name="bert-base-cased", num_labels=9)  # Assuming 9 NER tags
    prediction = model(inputs['input_ids'], inputs['attention_mask'])
    print("Hybrid model predictions shape:", prediction.shape)

def main():
    parser = argparse.ArgumentParser(description='Run NER models with CoNLL data')
    parser.add_argument('--train', default='eng.train', help='Path to training data')
    parser.add_argument('--test', default='eng.testb', help='Path to test data')
    parser.add_argument('--dev', default='eng.testa', help='Path to development data')
    parser.add_argument('--models', nargs='+', default=['hmm', 'crf', 'bert', 'gpt', 'hybrid'],
                        help='Models to run (hmm, crf, bert, gpt, hybrid)')
    args = parser.parse_args()
    
    # Preprocess data
    train_sentences, test_sentences, dev_sentences = preprocess_data(args.train, args.test, args.dev)
    
    # Run selected models
    model_funcs = {
        'hmm': lambda: run_hmm(train_sentences, test_sentences),
        'crf': lambda: run_crf(train_sentences, test_sentences),
        'bert': lambda: run_bert(test_sentences),
        'gpt': lambda: run_gpt(test_sentences),
        'hybrid': lambda: run_hybrid(test_sentences)
    }
    
    for model in args.models:
        if model.lower() in model_funcs:
            try:
                model_funcs[model.lower()]()
            except Exception as e:
                print(f"Error running {model} model: {str(e)}")
        else:
            print(f"Unknown model: {model}")

if __name__ == "__main__":
    main()
