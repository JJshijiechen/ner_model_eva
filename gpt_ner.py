# gpt_ner.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re

class GPTNER:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Define entity types for consistent output
        self.entity_types = {
            'PER': 'Person',
            'ORG': 'Organization',
            'LOC': 'Location',
            'MISC': 'Miscellaneous'
        }

    def create_prompt(self, text):
        """Create a structured prompt for NER task."""
        return f"""Extract named entities from the following sentence. For each entity, provide:
1. The entity text
2. The entity type (Person, Organization, Location, or Miscellaneous)

Sentence: "{text}"

Entities:
"""

    def parse_output(self, output):
        """Parse the model's output into a list of entities."""
        entities = []
        # Look for patterns like "text (TYPE)" or "text - TYPE"
        patterns = [
            r'([^()]+)\s*\(([^)]+)\)',  # Matches "text (TYPE)"
            r'([^-]+)\s*-\s*([^-]+)'    # Matches "text - TYPE"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, output)
            for match in matches:
                entity_text = match.group(1).strip()
                entity_type = match.group(2).strip().upper()
                # Map to standard NER tags
                if entity_type in ['PERSON', 'PER']:
                    tag = 'B-PER'
                elif entity_type in ['ORG', 'ORGANIZATION']:
                    tag = 'B-ORG'
                elif entity_type in ['LOC', 'LOCATION']:
                    tag = 'B-LOC'
                elif entity_type in ['MISC', 'MISCELLANEOUS']:
                    tag = 'B-MISC'
                else:
                    tag = 'O'
                
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'tag': tag
                })
        
        return entities

    def predict(self, text):
        """
        Predict named entities in the given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Dict]: List of detected entities with their types and positions
        """
        # Create structured prompt
        prompt = self.create_prompt(text)
        
        # Truncate the prompt if it's too long
        max_length = 512  # GPT-2's maximum context length
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        # Generate with a reasonable max_new_tokens
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,  # Increased for more detailed responses
            do_sample=True,
            top_p=0.95,
            temperature=0.7,  # Added temperature for more controlled generation
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode and parse the output
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        entities = self.parse_output(raw_output)
        
        return entities

if __name__ == "__main__":
    gpt_ner = GPTNER()
    sample_text = "John Doe is working at Acme Corp in New York City."
    result = gpt_ner.predict(sample_text)
    print("GPT NER output:")
    for entity in result:
        print(f"{entity['text']} ({entity['type']}) -> {entity['tag']}")
