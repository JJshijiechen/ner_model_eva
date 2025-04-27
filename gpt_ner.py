# gpt_ner.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPTNER:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def predict(self, prompt):
        """
        prompt: a string prompt instructing the model to extract entities.
        Returns: the generated text string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=100, do_sample=True, top_p=0.95, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    print("Sample sentence for GPT NER:", sample_sentence)
    sample_prompt = (f"Extract the named entities from the following sentence: '{sample_sentence}'. "
                     "List them in the format: ENTITY (LABEL)")
    gpt_ner = GPTNER()
    generated_text = gpt_ner.predict(sample_prompt)
    print("GPT NER generated output:")
    print(generated_text)