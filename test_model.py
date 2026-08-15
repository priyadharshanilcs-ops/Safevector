import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True
)

print("Model loaded successfully!")

prompt = "Explain what cloud security is."

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("Prompt:", prompt)
print("Number of hidden-state layers:", len(outputs.hidden_states))
print("Final hidden-state shape:", outputs.hidden_states[-1].shape)
final_layer = outputs.hidden_states[-1]

last_token_vector = final_layer[:, -1, :]

print("Extracted vector shape:", last_token_vector.shape)
print("First 10 vector values:")
print(last_token_vector[0][:10])
