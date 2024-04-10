import torch
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from preprocess import load_data, tokenize_and_format
from lion_pytorch import Lion
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification

from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

def train(model, train_data, val_data, optimizer, criterion, device, epochs):
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
                outputs = (outputs > 0.5).cpu().numpy().astype(float)

                all_val_preds.append(outputs)
                all_val_labels.append(labels.cpu().numpy())

            all_val_preds = np.concatenate(all_val_preds, axis=0)
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            # print(f'all_val_preds: {all_val_preds[:5]}, all_val_labels: {all_val_labels[:5]}')
            # print(f'all_val_preds: {np.sum(all_val_preds, axis=0)}, all_val_labels: {np.sum(all_val_labels, axis=0)}')
            accuracy = accuracy_score(all_val_labels, all_val_preds)

            if val_loss/len(val_data) < min_val_loss:
                min_val_loss = val_loss/len(val_data)
                # print(f'Saving model at epoch {epoch+1}, min_val_loss: {min_val_loss:.4f}')
                torch.save(model.state_dict(), 'best_model.pt')

        tqdm.write(f'Epoch [{epoch+1}/{epochs}], '
                   f'Loss: {total_loss / len(train_data):.4f}, '
                   f'Validation Loss: {val_loss/len(val_data):.4f}, '
                   f'Validation Accuracy: {accuracy:.4f}')
        


def calculate_pos_weight(train_data):
    positive_count = sum([label for _, _, label in train_data])
    total_count = len(train_data)
    negative_count = total_count - positive_count
    return negative_count.float() / positive_count.float()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'microsoft/deberta-v3-base'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_data = load_data('train.json')
    val_data = load_data('updated_val.json')

    processed_train = tokenize_and_format(train_data, tokenizer, dataset_type='train')
    processed_val = tokenize_and_format(val_data, tokenizer, dataset_type='val')

    epochs = 8

    # optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    optimizer = Lion(model.parameters(), lr=1e-6, weight_decay=5e-2)


    # 轉換為 Dataset
    train_dataset = CustomDataset(processed_train)
    val_dataset = CustomDataset(processed_val)

    pos_weight = calculate_pos_weight(train_dataset)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # 設定 batch size
    batch_size = 8  # 或者您希望的其他數值

    # 創建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 呼叫訓練函數
    train(model, train_loader, val_loader, optimizer, criterion, device, epochs)
