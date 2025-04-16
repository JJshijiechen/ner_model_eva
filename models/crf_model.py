# crf_model.py
import sklearn_crfsuite
from sklearn.metrics import classification_report

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
        X_train: list of lists, where each inner list is a sequence of feature dictionaries.
        y_train: list of lists, where each inner list is the corresponding sequence of labels.
        """
        self.crf.fit(X_train, y_train)
        print("CRF training complete.")

    def predict(self, X):
        """
        X: list of lists (each inner list corresponding to feature dictionaries for a sentence)
        Returns: list of predicted label sequences.
        """
        return self.crf.predict(X)

###################################
# Helper functions for data processing
###################################

def read_conll(filename):
    """
    Reads a CoNLL formatted file where each line contains a token and its label,
    and sentences are separated by blank lines.
    Returns a list of (tokens, labels) tuples.
    """
    sentences = []
    tokens = []
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                    tokens = []
                    labels = []
            else:
                parts = line.split()
                tokens.append(parts[0])
                labels.append(parts[-1])
        if tokens:
            sentences.append((tokens, labels))
    return sentences

def word2features(sent, i):
    """
    Extract features for the i-th token in a sentence.
    """
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features

def sentence2features(sent):
    """
    Convert a sentence into a list of feature dictionaries for each token.
    """
    return [word2features(sent, i) for i in range(len(sent))]

def prepare_dataset(sentences):
    """
    Convert the list of sentences into features (X) and labels (y) for the CRF model.
    """
    X = [sentence2features(tokens) for tokens, labels in sentences]
    y = [labels for tokens, labels in sentences]
    return X, y

###################################
# Main routine for CRF model training and evaluation
###################################
if __name__ == "__main__":
    train_file = "eng.train"
    test_file = "eng.testa"

    print("Reading training data from", train_file)
    train_sentences = read_conll(train_file)
    X_train, y_train = prepare_dataset(train_sentences)
    print(f"Number of training sentences: {len(train_sentences)}")

    print("Reading test data from", test_file)
    test_sentences = read_conll(test_file)
    X_test, y_test = prepare_dataset(test_sentences)
    print(f"Number of test sentences: {len(test_sentences)}")

    crf_ner = CRFNER()
    print("Training CRF model...")
    crf_ner.train(X_train, y_train)

    print("Predicting on test data...")
    y_pred = crf_ner.predict(X_test)
    print("CRF model evaluation:")
    print(classification_report(sum(y_test, []), sum(y_pred, [])))