from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
print("Loading AI model...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Simple example 
text = "The quick brown fox jumped over the lazy dog. The moment was" 
print(f"\nStarting text: '{text}'")
print("\nGenerating word by word...\n")

# Generate 5 words, one at a time
current_text = text
for step in range(30):
    print(f"--- Step {step + 1} ---")
    print(f"Current: '{current_text}'")
    
    # Encode current text
    input_ids = tokenizer.encode(current_text, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits
    
    # Get the predictions for the NEXT token
    next_token_logits = predictions[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    # Get top 5 predictions (changed 5 to 3)
    top_probs, top_indices = torch.topk(probs, 3)
    
    print("Top 3 next word predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        word = tokenizer.decode([idx])
        print(f"  {i+1}. '{word}' ({prob.item()*100:.1f}% confident)")
    
    # Use the most likely word --> Changed to sample using topk instead
    sampled_idx = torch.multinomial(top_probs, 1)
    next_token_id = top_indices[sampled_idx].item()

    next_word = tokenizer.decode([next_token_id])
    current_text += next_word
    
    print(f"✓ Chosen: '{next_word}'")
    print(f"New text: '{current_text}'\n")


print(f"\nFinal generated text: '{current_text}'")
