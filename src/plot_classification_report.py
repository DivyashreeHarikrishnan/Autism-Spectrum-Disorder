import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load classification report CSV
df = pd.read_csv("models/classification_report.csv", index_col=0)

# Drop accuracy/support rows for cleaner visual
df_plot = df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

plt.figure(figsize=(10, 6))
sns.heatmap(df_plot, annot=True, cmap="Blues", fmt=".2f")

plt.title("Classification Report Summary")
plt.xlabel("Metrics")
plt.ylabel("Classes")

plt.tight_layout()
plt.savefig("models/classification_report_plot.png")
print("Saved: models/classification_report_plot.png")
