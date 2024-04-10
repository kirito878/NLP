import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from preprocess import load_data, tokenize_and_format
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids'].squeeze()
        attention_mask = self.data[idx]['attention_mask'].squeeze()
        return input_ids, attention_mask

def predict(model, test_data, device):
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(test_data):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask).logits
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
            predictions.append(preds)

    predictions = np.concatenate(predictions, axis=0)
    return predictions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = 'microsoft/deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载训练好的模型
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()


    # 加载并处理测试数据
    test_data = load_data('updated_test.json')
    processed_test = tokenize_and_format(test_data, tokenizer, dataset_type='test')

    # 转换为 Dataset
    test_dataset = CustomDataset(processed_test)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 进行预测
    test_preds = predict(model, test_loader, device)

    # 创建 submission 文件
    submission = pd.DataFrame(test_preds, columns=['response_quality'])
    submission.to_csv('submission.csv', index=True, index_label='index')
