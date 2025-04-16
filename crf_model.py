# crf_model.py
import sklearn_crfsuite
from TorchCRF import CRF

class CRFNER:
    def __init__(self):
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def train(self, X_train, y_train):
        """
        X_train: list of lists, where each inner list is a sequence of feature dicts.
        y_train: list of lists, where each inner list is the corresponding sequence of labels.
        """
        self.crf.fit(X_train, y_train)
        print("CRF training complete.")

    def predict(self, X):
        """
        X: list of lists (each inner list corresponding to feature dicts for a sentence)
        Returns: list of predicted label sequences.
        """
        return self.crf.predict(X)

if __name__ == "__main__":
    # Example: Dummy features and labels for two sentences.
    X_train = [
        [{"word.lower()": "john", "is_title": True}, {"word.lower()": "doe", "is_title": True}],
        [{"word.lower()": "acme", "is_title": True}, {"word.lower()": "corp", "is_title": True}]
    ]
    y_train = [
        ["B-PER", "I-PER"],
        ["B-ORG", "I-ORG"]
    ]
    
    crf_ner = CRFNER()
    crf_ner.train(X_train, y_train)
    predictions = crf_ner.predict(X_train)
    print("CRF predicted labels:", predictions)
