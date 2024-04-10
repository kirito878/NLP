import json
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from lion_pytorch import Lion

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx][0]['input_ids'].squeeze()
        attention_mask = self.data[idx][0]['attention_mask'].squeeze()
        labels = self.data[idx][1]
        return input_ids, attention_mask, labels
    
class CustomTestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx][0]['input_ids'].squeeze()
        attention_mask = self.data[idx][0]['attention_mask'].squeeze()
        return input_ids, attention_mask

def update_data(file_name, all_gold_pred):
    # Load the existing data
    data = load_data(file_name)

    # Modify the data
    for i, item in enumerate(data):
        item['s.gold.index.predict'] = all_gold_pred[i]

    updated_file_name = './dataset/updated_' + file_name
    with open(updated_file_name, 'w') as file:
        json.dump(data, file, indent=4)

def load_data(file_name):
    file_path = os.path.join('dataset', file_name)
    print(f'Loading data from {file_path}')
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def preprocess_text(text):
    return text.strip().lower().replace('[', '').replace(']', '')

def tokenize_and_format(data, tokenizer, max_length=128):
    processed_data = []

    for i, item in tqdm(enumerate(data)):
        utterance = preprocess_text(item['u'])
        situational_statements = [preprocess_text(statement) for statement in item['s']]
        gold_index = item.get('s.gold.index', [])
        response = preprocess_text(item['r'])
        response_label = item.get('r.label', None)
        for index, statement in enumerate(situational_statements):
            formatted_text = f'{utterance} {response} {statement}'
            if index in gold_index:
                label_tensor = torch.tensor(1, dtype=torch.float)
            else:
                label_tensor = torch.tensor(0, dtype=torch.float)

            tokenized_text = tokenizer.encode_plus(
                formatted_text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )

            processed_data.append((tokenized_text, label_tensor))

    return processed_data


def train(model, train_data, val_data, test_data, optimizer, criterion, device, epochs):
    model.to(device)

    min_val_loss = 10000
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for input_ids, attention_mask, labels in tqdm(train_data):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.unsqueeze(1).to(device)

            model.zero_grad()

            # 執行模型
            outputs = model(input_ids, attention_mask=attention_mask).logits
            
            # print(f'outputs: {outputs.shape}, labels: {labels.shape}')

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # print(f'outputs: {outputs}, labels: {labels}')
        # 計算驗證集的損失
        val_loss = 0
        accuracy = 0
        all_val_preds = []
        all_val_labels = []
        all_gold_pred_val = []
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_data:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.unsqueeze(1).to(device)

                # print(f'input_ids: {input_ids}, attention_mask: {attention_mask}, labels: {labels}')
                # exit(0)
                outputs = model(input_ids, attention_mask=attention_mask).logits
                val_loss += criterion(outputs, labels)

                # 計算驗證集的準確率, if accuracy > 0.5, then 1, else 0
                # print(f'outputs: {outputs}, labels: {labels}')
                outputs = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)

                gold_pred = [i for i, output in enumerate(outputs) if output == 1]
                all_gold_pred_val.append(gold_pred)

                all_val_preds.append(outputs)
                all_val_labels.append(labels.cpu().numpy())

            all_val_preds = np.concatenate(all_val_preds, axis=0)
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            accuracy = accuracy_score(all_val_labels, all_val_preds)

            if val_loss/len(val_data) < min_val_loss:
                min_val_loss = val_loss/len(val_data)
                # print(f'Saving model at epoch {epoch+1}, min_val_loss: {min_val_loss:.4f}')
                torch.save(model.state_dict(), 'best_index_model.pt')

        tqdm.write(f'Epoch [{epoch+1}/{epochs}], '
                   f'Loss: {total_loss / len(train_data):.4f}, '
                   f'Validation Loss: {val_loss/len(val_data):.4f}, '
                   f'Validation Accuracy: {accuracy:.4f}')
    
    with torch.no_grad():
        all_gold_pred_test = []
        for input_ids, attention_mask in test_data:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask).logits

            outputs = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)

            gold_pred = [i for i, output in enumerate(outputs) if output == 1]

            all_gold_pred_test.append(gold_pred)
    
    return all_gold_pred_val, all_gold_pred_test

def calculate_pos_weight(train_data):
    positive_count = sum([label for _, _, label in train_data])
    total_count = len(train_data)
    negative_count = total_count - positive_count
    print(f'positive_count: {positive_count}, negative_count: {negative_count}')
    return negative_count.float() / positive_count.float()

if __name__ == "__main__":
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Adjust the file paths as necessary
    train_data = load_data('train.json')
    val_data = load_data('val.json')
    test_data = load_data('test.json')

    processed_train = tokenize_and_format(train_data, tokenizer)
    processed_val = tokenize_and_format(val_data, tokenizer)
    processed_test = tokenize_and_format(test_data, tokenizer)

    # # Show some examples
    # for i, (tokenized_text, label) in enumerate(processed_train):
    #     # Convert the tokenized text back to readable text
    #     readable_text = tokenizer.decode(tokenized_text['input_ids'][0])

    #     print(f"Example {i+1}:")
    #     print("Readable Text:", readable_text)
    #     print("Label:", label)
    #     print("------")

    #     # Limit to printing only the first few examples
    #     if i >= 2:  # Adjust this number to print more or fewer examples
    #         break

    epochs = 1

    # optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    optimizer = Lion(model.parameters(), lr=1e-6, weight_decay=5e-2)

    # 轉換為 Dataset
    train_dataset = CustomDataset(processed_train)
    val_dataset = CustomDataset(processed_val)
    test_dataset = CustomTestDataset(processed_test)

    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 設定 batch size
    batch_size = 12  # 或者您希望的其他數值
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 創建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 呼叫訓練函數
    all_gold_pred_val, all_gold_pred_test = train(model, train_loader, val_loader, test_loader, optimizer, criterion, device, epochs)

    print(f'all_gold_pred_val: {all_gold_pred_val[0]}')
    print(f'all_gold_pred_test: {all_gold_pred_test[0]}')

    # write to json file, add new key: s.gold.index.predict
    update_data('val.json', all_gold_pred_val)
    update_data('test.json', all_gold_pred_test)
