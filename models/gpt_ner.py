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

if __name__ == "__main__":
    gpt_ner = GPTNER()
    sample_prompt = ("Extract the named entities from the following sentence: "
                     "'John Doe works at Acme Corp in New York City.' "
                     "Entities:")
    generated_text = gpt_ner.predict(sample_prompt)
    print("GPT NER generated output:")
    print(generated_text)
