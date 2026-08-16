import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading instruction-tuned model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

print("Model loaded successfully!")

prompt = "Explain why multi-factor authentication improves security."

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

print("\nNumber of hidden-state layers:")
print(len(outputs.hidden_states))

print("\nFinal hidden-state shape:")
print(outputs.hidden_states[-1].shape)

vector = outputs.hidden_states[-1][:, -1, :]

print("\nExtracted vector shape:")
print(vector.shape)