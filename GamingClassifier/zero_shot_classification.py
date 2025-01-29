import pandas as pd
from transformers import pipeline

# Load the sampled dataset
file_path = "reddit_gaming_sample.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = ["gaming-related", "not gaming-related"]

# Classify comments
def classify_comment(comment):
    result = classifier(comment, candidate_labels)
    return 1 if result["labels"][0] == "gaming-related" else 0

# Apply classification
df["gaming_related"] = df["Comment"].apply(classify_comment)

# Save labeled dataset
df.to_csv("reddit_gaming_sample_labeled.csv", index=False)
print("Classification completed! Labeled file saved as 'reddit_gaming_sample_labeled.csv'.")
