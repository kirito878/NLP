# %%
from torch.optim import AdamW
import csv
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import random
import time

# %%
train_data = pd.read_csv('dataset/train.tsv', sep='\t')
val_data = pd.read_csv('dataset/val.tsv', sep='\t')
test_data = pd.read_csv('dataset/test.tsv', sep='\t')
train_data = pd.concat([train_data, val_data], axis=0)
# train_data

# %%
num_classes = 6
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tag_list = ["AM", "MS", "OTHER", "PH", "SF", "SR"]
tag_to_idx = {}
for i in range(len(tag_list)):
    tag_to_idx[tag_list[i]] = i


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utterance = str(self.data.iloc[idx]['utterance'])
        labels = self.data.iloc[idx]['classes']
        labels = self.label_encoded(labels)
        encoding = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

    def label_encoded(self, labels):
        encode_label = np.zeros(6, dtype=int)
        list_label = labels.split(",")
        for i in list_label:
            idx = tag_to_idx[i]
            encode_label[idx] = 1
        return encode_label


max_len = 256
batch_size = 8

train_dataset = CustomDataset(train_data, tokenizer, max_len)
val_dataset = CustomDataset(val_data, tokenizer, max_len)


train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# %%

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %%


def set_model(model, epochs):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5,
                      eps=1e-6, betas=(0.9, 0.999))
    total_steps = len(train_dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=total_steps)
    return model, optimizer, scheduler


# %%
weights = [1, 1, 0.2, 1, 1, 1]
weights = torch.tensor(weights).to(device)
loss_fn = nn.BCEWithLogitsLoss()


def evalute(dataloader, model):
    model.eval()
    y_pred = []
    y_target = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            b_input_ids = batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            logits = model(b_input_ids, b_attn_mask)
            y_pred.extend(torch.sigmoid(
                logits.logits).cpu().detach().numpy().tolist())
            y_target.extend(b_labels.cpu().detach().numpy().tolist())
    y_preds = (np.array(y_pred) > 0.5).astype(int)
    marco_f1 = f1_score(y_target, y_preds, average='macro')
    # print("marco f1 score : ",marco_f1)
    return marco_f1


def train(model, train_dataloader, val_dataloader, optimizer, scheduler, path, epochs, evaluation):
    max_score = 0
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'F1 score':^9} | {'Elapsed':^9}")
    for epoch_i in range(epochs):
        t0_epoch = time.time()
        total_loss = 0
        for _, batch in enumerate(train_dataloader):
            model.train()
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        if evaluation == True:
            time_elapsed = time.time() - t0_epoch
            score = evalute(val_dataloader, model)
            if score > max_score:
                torch.save(model.state_dict(), path)
                # print('save model')
                max_score = score
            print(
                f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {score:^9.6f} | {time_elapsed:^9.2f}")
        print("")
    print("best score: ", max_score)


# %%
model_list = []
for i in range(1):
    epochs = 10
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes)
    bert_classifier, optimizer, scheduler = set_model(model, epochs=epochs)
    path = f"w_weight{i}.pth"
    train(bert_classifier, train_dataloader, val_dataloader,
          optimizer, scheduler, path, epochs=epochs, evaluation=True)
    model_list.append(model)

# %%


def essem_evalute(dataloader, model_list):
    for i in range(len(model_list)):
        model_list[i].eval()
    y_pred = []
    y_target = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            b_input_ids = batch['input_ids'].to(device)
            b_attn_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            tmp = 0
            for i in range(len(model_list)):
                logits = model_list[i](b_input_ids, b_attn_mask)
                tmp += torch.sigmoid(logits.logits)
            tmp = tmp / len(model_list)
            y_pred.extend(tmp.cpu().detach().numpy().tolist())
            y_target.extend(b_labels.cpu().detach().numpy().tolist())
    y_preds = (np.array(y_pred) > 0.5).astype(int)
    marco_f1 = f1_score(y_target, y_preds, average='macro')
    # print("marco f1 score : ",marco_f1)
    return marco_f1


print(essem_evalute(val_dataloader, model_list))

# %%


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data.iloc[idx]['id']
        utterance = str(self.data.iloc[idx]['utterance'])
        encoding = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'id': ids,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


test_dataset = TestDataset(test_data, tokenizer, max_len)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %%
prdict_model_list = []
for i in range(len(model_list)):
    path = f'./w_weight{i}.pth'
    bert_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes).to(device)
    bert_classifier.load_state_dict(torch.load(path))
    prdict_model_list.append(bert_classifier)
    bert_classifier.eval()
y_pred = []
ids = []
with torch.no_grad():
    for batch in test_dataloader:
        id_0 = batch['id'].cpu().item()
        ids.append(id_0)
        b_input_ids = batch['input_ids'].to(device)
        b_attn_mask = batch['attention_mask'].to(device)
        tmp = 0
        for i in range(len(prdict_model_list)):
            logits = prdict_model_list[i](b_input_ids, b_attn_mask)

            tmp += torch.sigmoid(logits.logits)
        tmp = tmp / len(prdict_model_list)
        y_pred.extend(tmp.cpu().detach().numpy().tolist())

# %%
y_preds = (np.array(y_pred) > 0.6).astype(int)
# y_preds

# %%
df = pd.DataFrame(y_preds, columns=tag_list)
df_id = pd.DataFrame(ids, columns=["id"])
merged_df = pd.concat([df_id, df], axis=1)
# merged_df

# %%
data_rows = merged_df.to_dict(orient='records')

# %%
with open("submission.csv", 'w', newline='') as csvfile:
    fieldnames = ["id", "AM", "MS", "OTHER", "PH", "SF", "SR"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data_rows)
