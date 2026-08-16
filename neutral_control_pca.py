import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


model_name = "distilgpt2"
selected_layer = 4

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

print("Model loaded successfully!")


group_a = [
    "Explain the purpose of access control.",
    "Explain how a firewall protects a network.",
    "Explain the purpose of multi-factor authentication.",
    "Explain why encryption is important.",
    "Explain how cloud security protects data.",
    "Explain why strong passwords are important.",
    "Explain the purpose of network security.",
    "Explain why software updates improve security.",
    "Explain how least privilege improves security.",
    "Explain how organizations protect sensitive data."
]


group_b = [
    "Please explain the purpose of access control.",
    "Please explain how a firewall protects a network.",
    "Please explain the purpose of multi-factor authentication.",
    "Please explain why encryption is important.",
    "Please explain how cloud security protects data.",
    "Please explain why strong passwords are important.",
    "Please explain the purpose of network security.",
    "Please explain why software updates improve security.",
    "Please explain how least privilege improves security.",
    "Please explain how organizations protect sensitive data."
]


def get_vector(prompt):

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
        )

    hidden_state = outputs.hidden_states[selected_layer]
    vector = hidden_state[:, -1, :]

    return vector.squeeze(0).cpu().numpy()


print("Extracting Group A vectors...")
a_vectors = [get_vector(p) for p in group_a]

print("Extracting Group B vectors...")
b_vectors = [get_vector(p) for p in group_b]


all_vectors = np.array(a_vectors + b_vectors)

print("Original vector shape:")
print(all_vectors.shape)

all_vectors = normalize(all_vectors)

pca = PCA(n_components=2)
pca_vectors = pca.fit_transform(all_vectors)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)


a_pca = pca_vectors[:10]
b_pca = pca_vectors[10:]


a_center = a_pca.mean(axis=0)
b_center = b_pca.mean(axis=0)

distance = np.linalg.norm(a_center - b_center)

print("\nGroup A center:")
print(a_center)

print("\nGroup B center:")
print(b_center)

print("\nDistance between neutral group centers:")
print(distance)


plt.figure(figsize=(8, 6))

plt.scatter(
    a_pca[:, 0],
    a_pca[:, 1],
    label="Normal Safe",
    marker="o"
)

plt.scatter(
    b_pca[:, 0],
    b_pca[:, 1],
    label="Please + Safe",
    marker="x"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Neutral Control PCA - Layer 4")
plt.legend()
plt.grid(True)

plt.savefig(
    "neutral_control_layer4.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved as neutral_control_layer4.png")