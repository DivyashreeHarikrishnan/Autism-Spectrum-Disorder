import matplotlib.pyplot as plt

# Model metrics
metrics = {
    "Accuracy": 0.5450,
    "Precision": 0.7368,
    "Recall": 0.1400,
    "F1-Score": 0.2353,
    "ROC-AUC": 0.4948
}

# Extract keys and values
names = list(metrics.keys())
values = list(metrics.values())

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(names, values)

plt.title("Model Performance Comparison")
plt.ylabel("Scores")
plt.ylim(0, 1)  # scale 0–1
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Show plot
plt.show()
    