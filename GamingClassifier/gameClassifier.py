#  BERT (Bidirectional Encoder Representations from Transformers) for this task. It’s widely accessible, performs well for classification tasks, and has excellent support through libraries like Hugging Face's transformers.

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the CSV
df = pd.read_csv("reddit_gaming_sample_labeled.csv")

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)
dataset = dataset.rename_column("Comment", "text")
dataset = dataset.rename_column("gaming_related", "label")

# label_mapping = {
#     "negative": 0,
#     "neutral": 1,
#     "positive": 2
# }
# def preprocess_labels(data_set):
#     # Convert the string label to its corresponding integer
#     data_set["label"] = label_mapping[data_set["label"]]
#     return data_set

# Apply the transformation
# dataset = dataset.map(preprocess_labels)
dataset = dataset.train_test_split(test_size=0.2)  # Split into train/test



model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels: gaming-related or not //////!!!!! this mismatches the labels

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])  # Remove raw text column
tokenized_dataset.set_format("torch")


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",          # Where to save results
    evaluation_strategy="epoch",    # Evaluate after each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,              # Regularization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

from sklearn.metrics import classification_report

predictions = trainer.predict(tokenized_dataset["test"])
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

print(classification_report(labels, preds))

text = "Do you have any recommendations for RPG games?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1)
print("Gaming-related" if prediction == 1 else "Not gaming-related")