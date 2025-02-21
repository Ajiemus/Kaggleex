import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, AutoTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
from IPython import embed

# CSV-Dateien laden
train_df = pd.read_csv('train.csv')
predict_df = pd.read_csv('test.csv')

# Tokenizer und Modell laden
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Training und Validierung Daten aufteilen
X= train_df['text'].tolist()
Y=train_df['target'].tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(
    X, Y, test_size=0.2
)
embed()
# Tokenisierung
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
predict_encodings = tokenizer(predict_df['text'].tolist(), truncation=True, padding=True, max_length=512)

# Dataset-Objekte erstellen
class DisasterDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = DisasterDataset(train_encodings, train_labels)
val_dataset = DisasterDataset(val_encodings, val_labels)
predict_dataset = DisasterDataset(predict_encodings)

# DataLoader erstellen
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
predict_loader = DataLoader(predict_dataset, batch_size=8, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(3):  # Anzahl der Epochen
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Training loss: {train_loss / len(train_loader)}")

# Validation Loop
model.eval()
val_loss = 0
for batch in tqdm(val_loader, desc="Validation"):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        val_loss += loss.item()
print(f"Validation loss: {val_loss / len(val_loader)}")

# Vorhersagen machen
model.eval()
predictions = []
for batch in tqdm(predict_loader, desc="Prediction"):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# Ergebnisse in die CSV-Datei schreiben
predict_df['label'] = predictions
predict_df['label'] = predict_df['label'].apply(lambda x: 'Disaster' if x == 1 else 'Not Disaster')
predict_df.to_csv('submission.csv', index=False)

print("Vorhersagen wurden in die Datei 'submission.csv' geschrieben.")
