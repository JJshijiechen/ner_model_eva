# bert_ner.py
from transformers import pipeline

class BERTNER:
    def __init__(self):
        # You can choose any model fine-tuned for NER, e.g., "dbmdz/bert-large-cased-finetuned-conll03-english"
        # this models is:
        # Pre-trained on large English corpus
        # Fine-tuned on CoNLL-2003 NER dataset
        # Cased version (preserves capitalization, important for NER)
        # https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english
        self.nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

    def predict(self, text):
        """
        text: a string input.
        Returns: list of entities with their labels and positions.
        """
        return self.nlp(text)

if __name__ == "__main__":
    bert_ner = BERTNER()
    sample_text = "John Doe is working at Acme Corp in New York City."
    results = bert_ner.predict(sample_text)
    print("BERT NER results:")
    for entity in results:
        print(entity)
