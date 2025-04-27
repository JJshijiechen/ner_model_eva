# bert_ner.py
from transformers import pipeline

class BERTNER:
    def __init__(self):
        # Using a model fine-tuned on CoNLL-2003 NER
        # Set aggregation_strategy to "simple" for full‐span grouping
        self.nlp = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )

    def predict(self, text):
        """
        text: a string input.
        Returns: predictions as a list of dicts with keys:
          - 'word' (the span text)
          - 'entity_group' (the label)
          - 'start','end' (char offsets)
        """
        return self.nlp(text)


def read_conll_sentences(filename):
    """
    Reads a file in CoNLL format and extracts sentences.
    Each sentence is constructed by taking only the first token of each non-empty line.
    A blank line indicates a sentence break.
    
    Returns:
        A list of sentences (each sentence is a single string with tokens separated by a space).
    """
    sentences = []
    tokens = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # Blank line signals end of sentence.
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
    # Set the file path (adjust if necessary)
    test_file = "eng.testa"
    sentences = read_conll_sentences(test_file)
    bert_ner = BERTNER()
    
    print("BERT NER results on test sentences:")
    # Process first 5 sentences as a demo.
    for i, sentence in enumerate(sentences[:5]):
        predictions = bert_ner.predict(sentence)
        print(f"\nSentence {i+1}: {sentence}")
        # For each span prediction, print each token in the span with its label
        # (or just show the span as a whole if you prefer).
        for token_pred in predictions:
            word  = token_pred.get('word')
            label = token_pred.get('entity') or token_pred.get('entity_group')
            print(f"{word} {label}")
        # Blank line between sentences.
        print("") 