# hybrid_model.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

class BERT_CRF(nn.Module):
    def __init__(self, bert_model_name="bert-base-cased", num_labels=5):
        """
        num_labels: number of NER labels (e.g., O, B-PER, I-PER, B-ORG, I-ORG, etc.)
        """
        super(BERT_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        If labels are provided, compute the negative log-likelihood loss.
        Otherwise, decode the best path.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.hidden2tag(outputs.last_hidden_state)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction

def read_first_sentence(filename):
    """
    Reads the first non-empty sentence (tokens only) from a CoNLL file.
    """
    tokens = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    return " ".join(tokens)
            else:
                parts = line.split()
                tokens.append(parts[0])
    return " ".join(tokens)

if __name__ == "__main__":
    test_file = "eng.testa"
    sample_sentence = read_first_sentence(test_file)
    print("Sample sentence for Hybrid Model:", sample_sentence)
    
    # Initialize tokenizer and model.
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BERT_CRF(bert_model_name="bert-base-cased", num_labels=5)
    
    # Tokenize the sample sentence.
    inputs = tokenizer(sample_sentence, return_tensors="pt")
    # Inference (no labels provided).
    predictions = model(inputs['input_ids'], inputs['attention_mask'])
    
    print("Hybrid BERT+CRF predictions (token-level indices):")
    print(predictions)