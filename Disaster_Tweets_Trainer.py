import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset

# CSV-Dateien laden
train_df = pd.read_csv('train.csv')
predict_df = pd.read_csv('predict.csv')

# Tokenizer und Modell laden
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training und Validierung Daten aufteilen
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].tolist(), train_df['target'].tolist(), test_size=0.2
)

# Tokenisierung
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
predict_encodings = tokenizer(predict_df['text'].tolist(), truncation=True, padding=True)

# Dataset-Objekte erstellen
class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = DisasterDataset(train_encodings, train_labels)
val_dataset = DisasterDataset(val_encodings, val_labels)
predict_dataset = DisasterDataset(predict_encodings)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Modell trainieren
trainer.train()

# Vorhersagen machen
predictions = trainer.predict(predict_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

# Ergebnisse in die CSV-Datei schreiben
predict_df['label'] = predicted_labels
predict_df['label'] = predict_df['label'].apply(lambda x: 'Disaster' if x == 1 else 'Not Disaster')
predict_df.to_csv('predict.csv', index=False)

print("Vorhersagen wurden in die Datei 'predict.csv' geschrieben.")
