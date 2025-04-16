# bert_ner.py
from transformers import BertTokenizer, BertForTokenClassification
from typing import List, Dict, Any
import torch

class BERTNER:
    def __init__(self, model_name: str = "bert-base-cased"):
        """
        Initialize the BERT NER model.
        
        Args:
            model_name (str): Name of the pre-trained BERT model to use
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict named entities in the given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Dict[str, Any]]: List of detected entities with their types and positions
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        # Convert predictions to readable format
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = []
        
        # This is a simplified version - in practice, you'd want to implement
        # proper entity extraction and grouping
        for token, pred in zip(tokens, predictions[0]):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                entities.append({
                    "token": token,
                    "label": f"LABEL_{pred.item()}"
                })
                
        return entities

if __name__ == "__main__":
    bert_ner = BERTNER()
    sample_text = "John Doe is working at Acme Corp in New York City."
    results = bert_ner.predict(sample_text)
    print("BERT NER results:")
    for entity in results:
        print(entity)
