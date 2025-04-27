# hmm_model.py
import numpy as np
from hmmlearn import hmm

class HMMNER:
    def __init__(self, n_states=5, n_features=100):
        # n_states could represent your label types (e.g., O, PER, LOC, ORG, MISC)
        # n_features should match the vocabulary size.
        self.model = hmm.MultinomialHMM(n_components=n_states, n_iter=100)
        self.n_features = n_features

    def train(self, X, lengths):
        """
        X: 2D numpy array of shape (n_samples, 1) with discrete feature indices.
        lengths: list of sequence lengths.
        """
        self.model.fit(X, lengths)
        print("HMM training complete.")

    def predict(self, X):
        """
        X: 2D numpy array of shape (n_samples, 1)
        Returns: predicted state sequence.
        """
        return self.model.predict(X)

###################################
# Helper functions for data
###################################
def read_conll(filename):
    """
    Reads a CoNLL formatted file where each line contains a token and its label.
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

def build_vocab(sentences):
    """
    Build a vocabulary dictionary (word to index) from the training sentences.
    """
    vocab = {}
    for tokens, _ in sentences:
        for w in tokens:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab 

def prepare_hmm_data(sentences, vocab):
    """
    Convert sentences into a sequence of indices.
    Returns:
      X: 2D numpy array of shape (total_tokens, 1)
      lengths: list of sentence lengths.
      gold_labels: list of label sequences (for evaluation).
    """
    X_list = []
    lengths = []
    gold_labels = []
    for tokens, labels in sentences:
        seq = [vocab.get(w, -1) for w in tokens]  # unknown tokens mapped to -1
        X_list.extend([[idx] for idx in seq])
        lengths.append(len(tokens))
        gold_labels.append(labels)
    X = np.array(X_list)
    return X, lengths, gold_labels

def build_label_mapping(sentences):
    """
    Build a mapping of label to index from the training data.
    """
    label_set = set()
    for _, labels in sentences:
        label_set.update(labels)
    label_list = sorted(label_set)
    label2idx = {label: idx for idx, label in enumerate(label_list)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label

###################################
# Main routine for HMM model
###################################
if __name__ == "__main__":
    train_file = "eng.train"
    test_file = "eng.testa"

    print("Reading training data from", train_file)
    train_sentences = read_conll(train_file)
    print(f"Number of training sentences: {len(train_sentences)}")

    print("Reading test data from", test_file)
    test_sentences = read_conll(test_file)
    print(f"Number of test sentences: {len(test_sentences)}")

    # Build vocabulary and label mapping from training data.
    vocab = build_vocab(train_sentences)
    label2idx, idx2label = build_label_mapping(train_sentences)

    n_states = len(label2idx)
    n_features = len(vocab)
    
    # Prepare training data for HMM.
    X_train, lengths_train, _ = prepare_hmm_data(train_sentences, vocab)
    hmm_ner = HMMNER(n_states=n_states, n_features=n_features)
    print("Training HMM model...")
    hmm_ner.train(X_train, lengths_train)

    # Prepare test data.
    X_test, lengths_test, test_gold = prepare_hmm_data(test_sentences, vocab)
    pred_seq = hmm_ner.predict(X_test)
    
    # Map predicted states to labels using a dummy mapping (this is for demonstration only)
    pred_labels = []
    start = 0
    for l in lengths_test:
        seq = pred_seq[start:start+l]
        # Here we simply assume that state index corresponds to label index.
        pred_labels.append([ idx2label.get(state, "O") for state in seq ])
        start += l

    # Flatten gold and predicted labels for evaluation.
    gold_flat = sum([labels for _, labels in test_sentences], [])
    pred_flat = sum(pred_labels, [])
    print("HMM model evaluation (dummy state-label mapping):")
    try:
        from sklearn.metrics import classification_report
        print(classification_report(gold_flat, pred_flat))
    except Exception as e:
        print("Error in evaluation:", e) 