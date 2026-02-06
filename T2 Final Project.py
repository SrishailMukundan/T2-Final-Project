from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import random

# Load model
print("Loading AI model...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
temperature = 1.1

#Changed the prompt to help the model to predict the next words. Does not always rhyme but it works some of the time. 
start = "Write a short rhyming poem about a peaceful day:\nA calm and pleasant day it was,"
current_text = start

sentences = random.randint(4,6)

poem = 'A calm and pleasant day it was,' + '\n'

for i in range(sentences):
    words = random.randint(4,7)
    line = ""
    j=0
    attempts = 0
    while j < words:
        input = tokenizer.encode(current_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(input)
            predictions = outputs.logits

         # Get the predictions for the NEXT token
        next_token_logits = predictions[0, -1, :]
        probs = torch.softmax(next_token_logits/temperature, dim=-1)
        top_probs, top_indices = torch.topk(probs, 50)

        sampled_idx = torch.multinomial(top_probs, 1)
        next_token_id = top_indices[sampled_idx].item()

        word = tokenizer.decode(next_token_id, skip_special_tokens=True).strip()
        attempts = 0
        if not re.search(r'\w', word):  # keep letters/numbers
            if attempts > 5:  # fallback after 5 tries
                continue  # accept even if it’s a fragment
            continue

        line += ' ' + word
        current_text += ' ' + word
        j+=1
    poem += line + '\n'
    current_text += "\n"
    line = ""

print(poem)
