import csv

results = [
    ["Layer", "Average Similarity"],
    [0, 0.985956],
    [1, 0.993632],
    [2, 0.991349],
    [3, 0.985652],
    [4, 0.976891],
    [5, 0.981685],
    [6, 0.999084]
]

with open("controlled_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(results)

print("Results saved successfully to controlled_results.csv")