import csv
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
selected_layer = 21


print("Loading SmolLM2...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

print("Model loaded successfully!")


# Read dataset
rows = []

with open(
    "dataset.csv",
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        rows.append(row)


print("Total prompts loaded:", len(rows))


def get_vector(prompt):

    messages = [
        {
            "role": "system",
            "content": "Follow the system instructions and answer the user helpfully."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True
        )

    vector = outputs.hidden_states[
        selected_layer
    ][:, -1, :]

    return (
        vector
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )


vectors = []

print("\nExtracting Layer 21 vectors...")


for index, row in enumerate(rows):

    vector = get_vector(row["prompt"])

    vectors.append(vector)

    print(
        f"Processed {index + 1}/{len(rows)}"
    )


vectors = np.array(vectors)


print("\nVector extraction complete!")

print("Vector matrix shape:")
print(vectors.shape)


# Save vectors
np.save(
    "layer21_vectors.npy",
    vectors
)


# Save corresponding metadata
with open(
    "vector_metadata.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "topic_id",
            "topic",
            "prompt_type",
            "label",
            "prompt"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print("\nSaved:")
print("layer21_vectors.npy")
print("vector_metadata.csv")