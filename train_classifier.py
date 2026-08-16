import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# Load vectors
# --------------------------------------------------

vectors = np.load("layer21_vectors.npy")

metadata = []

with open(
    "vector_metadata.csv",
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        metadata.append(row)


print("Vector shape:", vectors.shape)
print("Metadata rows:", len(metadata))


# --------------------------------------------------
# Topic-wise split
# T01-T24 = training
# T25-T30 = testing
# --------------------------------------------------

train_indices = []
test_indices = []

for i, row in enumerate(metadata):

    topic_number = int(
        row["topic_id"].replace("T", "")
    )

    if topic_number <= 24:
        train_indices.append(i)

    else:
        test_indices.append(i)


X_train = vectors[train_indices]
X_test = vectors[test_indices]


# jailbreak_style = 1
# safe = 0

y_train = np.array([
    1 if metadata[i]["label"] == "jailbreak_style" else 0
    for i in train_indices
])

y_test = np.array([
    1 if metadata[i]["label"] == "jailbreak_style" else 0
    for i in test_indices
])


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Training safe:",
    np.sum(y_train == 0)
)

print(
    "Training jailbreak-style:",
    np.sum(y_train == 1)
)

print(
    "Testing safe:",
    np.sum(y_test == 0)
)

print(
    "Testing jailbreak-style:",
    np.sum(y_test == 1)
)


# --------------------------------------------------
# Build classifier
# --------------------------------------------------

classifier = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining classifier...")

classifier.fit(
    X_train,
    y_train
)

print("Training complete!")


# --------------------------------------------------
# Test
# --------------------------------------------------

predictions = classifier.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n==============================")
print("CLASSIFIER RESULTS")
print("==============================")

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Safe",
            "Jailbreak-style"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# Show individual predictions
# --------------------------------------------------

print("\nTEST PREDICTIONS")
print("------------------------------")

for index, prediction in zip(
    test_indices,
    predictions
):

    actual = metadata[index]["label"]

    predicted_label = (
        "jailbreak_style"
        if prediction == 1
        else "safe"
    )

    print(
        metadata[index]["topic_id"],
        "| Actual:",
        actual,
        "| Predicted:",
        predicted_label
    )


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Safe",
        "Jailbreak-style"
    ]
)

display.plot()

plt.title(
    "SafeVector - Layer 21 Classifier"
)

plt.savefig(
    "classifier_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nSaved classifier_confusion_matrix.png"
)