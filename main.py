# main.py
from hmm_model import HMMNER
from crf_model import CRFNER
from bert_ner import BERTNER
from gpt_ner import GPTNER
from hybrid_model import BERT_CRF
from transformers import BertTokenizer
import numpy as np
import torch

def run_hmm():
    print("Running HMM Model:")
    # Dummy data for demonstration
    X_train = np.random.randint(0, 100, size=(10, 1))
    lengths = [6, 4]
    model = HMMNER(n_states=5, n_features=100)
    model.train(X_train, lengths)
    pred = model.predict(X_train)
    print("HMM predictions:", pred)

def run_crf():
    print("\nRunning CRF Model:")
    X_train = [
        [{"word.lower()": "john", "is_title": True}, {"word.lower()": "doe", "is_title": True}],
        [{"word.lower()": "acme", "is_title": True}, {"word.lower()": "corp", "is_title": True}]
    ]
    y_train = [
        ["B-PER", "I-PER"],
        ["B-ORG", "I-ORG"]
    ]
    model = CRFNER()
    model.train(X_train, y_train)
    pred = model.predict(X_train)
    print("CRF predictions:", pred)

def run_bert():
    print("\nRunning BERT-based NER:")
    model = BERTNER()
    sample_text = "John Doe is working at Acme Corp in New York City."
    result = model.predict(sample_text)
    print("BERT NER output:")
    for entity in result:
        print(entity)

def run_gpt():
    print("\nRunning GPT-based NER (via prompt engineering):")
    model = GPTNER()
    prompt = ("Extract the named entities from the following sentence: "
              "'John Doe works at Acme Corp in New York City.' Entities:")
    result = model.predict(prompt)
    print("GPT NER output:")
    print(result)

def run_hybrid():
    print("\nRunning Hybrid BERT+CRF Model:")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    sample_text = "John Doe works at Acme Corp in New York City."
    inputs = tokenizer(sample_text, return_tensors="pt")
    model = BERT_CRF(bert_model_name="bert-base-cased", num_labels=5)
    # For demonstration, we are only doing decoding.
    prediction = model(inputs['input_ids'], inputs['attention_mask'])
    print("Hybrid model predictions:", prediction)

if __name__ == "__main__":
    run_hmm()
    run_crf()
    run_bert()
    run_gpt()
    run_hybrid()
