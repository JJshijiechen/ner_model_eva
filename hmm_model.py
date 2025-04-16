# hmm_model.py
import numpy as np
from hmmlearn import hmm

class HMMNER:
    def __init__(self, n_states=9, n_features=100):
        # n_states could represent your label types (e.g., O, PER, LOC, ORG, MISC)
        # n_features should match the dimensionality of your feature encoding
        self.model = hmm.CategoricalHMM(
            n_components=n_states,
            n_iter=100,
            tol=0.01,  # Convergence threshold
            verbose=True,
            params='ste',  # Only update startprob, transmat, and emissionprob
            init_params='ste',  # Initialize startprob, transmat, and emissionprob
            random_state=42
        )
        self.n_features = n_features
        self.n_states = n_states

    def train(self, X, lengths):
        """
        Train the HMM model.
        X: 2D numpy array of shape (n_samples, 1) with discrete feature indices.
        lengths: list of lengths of each sequence.
        """
        # Add small regularization to avoid degenerate solutions
        self.model.transmat_ = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        self.model.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.model.emissionprob_ = np.full((self.n_states, self.n_features), 1.0 / self.n_features)
        
        self.model.fit(X, lengths)
        print("HMM training complete.")

    def predict(self, X):
        """
        Predict state sequence for the given feature array.
        X: 2D numpy array of shape (n_samples, 1)
        Returns: predicted label sequence.
        """
        # Use hmmlearn's built-in prediction
        pred = self.model.predict(X)
        # Convert numeric predictions to string labels
        label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
                    5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
        return [label_map[p] for p in pred]

if __name__ == "__main__":
    # Example: Dummy training data.
    # Suppose we have 10 tokens (encoded as integers between 0 and n_features-1)
    # and two sentences of lengths 6 and 4.
    X_train = np.random.randint(0, 100, size=(10, 1))
    lengths = [6, 4]

    hmm_ner = HMMNER(n_states=5, n_features=100)
    hmm_ner.train(X_train, lengths)
    predictions = hmm_ner.predict(X_train)
    print("Predicted labels:", predictions)
