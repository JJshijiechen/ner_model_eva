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
            # The mask should be of type torch.uint8 in older versions or torch.bool in newer ones.
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction

if __name__ == "__main__":
    # Example usage with dummy data:
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    sample_text = "John Doe works at Acme Corp in New York City."
    inputs = tokenizer(sample_text, return_tensors="pt")
    
    # For testing, labels are not used. For training, you would provide labels.
    model = BERT_CRF(bert_model_name="bert-base-cased", num_labels=5)
    predictions = model(inputs['input_ids'], inputs['attention_mask'])
    print("Hybrid BERT+CRF predictions:", predictions)
