# bert_ner.py
from transformers import pipeline

class BERTNER:
    def __init__(self):
        # Using a model fine-tuned on CoNLL-2003 NER
        self.nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

    def predict(self, text):
        """
        text: a string input.
        Returns: list of entities with their labels and positions.
        """
        return self.nlp(text)

def read_conll_sentences(filename):
    """
    Reads a file in CoNLL format and extracts sentences (using only the first token in each line).
    """
    sentences = []
    tokens = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(" ".join(tokens))
                    tokens = []
            else:
                parts = line.split()
                tokens.append(parts[0])
        if tokens:
            sentences.append(" ".join(tokens))
    return sentences

if __name__ == "__main__":
    test_file = "eng.testa"
    sentences = read_conll_sentences(test_file)
    bert_ner = BERTNER()
    print("BERT NER results on test sentences:")
    for i, sentence in enumerate(sentences[:5]):  # Process first 5 sentences as a demo
        results = bert_ner.predict(sentence)
        print(f"\nSentence {i+1}: {sentence}")
        for entity in results:
            print(entity)