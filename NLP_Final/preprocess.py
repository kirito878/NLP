import json
import os
from transformers import BertTokenizer
import torch
from transformers import AutoTokenizer
from gold import get_gold_indices
from tqdm import tqdm

def load_data(file_name):
    file_path = os.path.join('dataset', file_name)
    print(f'Loading data from {file_path}')
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def preprocess_text(text):
    return text.strip().lower().replace('[', '').replace(']', '')

def tokenize_and_format(data, tokenizer, max_length=256, dataset_type='train'):
    processed_data = []

    for i, item in tqdm(enumerate(data)):
        utterance = preprocess_text(item['u'])
        situational_statements = [preprocess_text(statement) for statement in item['s']]
        # situational_types = item['s.type']
        response = preprocess_text(item['r'])
        response_label = item.get('r.label', None)

        if dataset_type != 'train':
            gold_index = item.get('s.gold.index.predict', [])
            combined_statements = ' '.join([f"{preprocess_text(statement)}"
                                            for index, statement in enumerate(situational_statements) if index in gold_index])
        else:
            gold_index = item.get('s.gold.index', [])
            combined_statements = ' '.join([f"{preprocess_text(statement)}"
                                            for index, statement in enumerate(situational_statements) if index in gold_index])
        
        formatted_text = f"{utterance} {combined_statements} {response}"

        # BERT tokenizer
        tokenized_text = tokenizer.encode_plus(
            formatted_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )


        if dataset_type == 'test':
            label_tensor = None
            processed_data.append(tokenized_text)
        else:
            label_tensor = torch.tensor(response_label, dtype=torch.float) if response_label is not None else None
            processed_data.append((tokenized_text, label_tensor))

    return processed_data


if __name__ == "__main__":
    model_name = 'microsoft/deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Adjust the file paths as necessary
    train_data = load_data('train.json')
    val_data = load_data('updated_val.json')

    processed_train = tokenize_and_format(train_data, tokenizer, dataset_type='train')
    processed_val = tokenize_and_format(val_data, tokenizer, dataset_type='val')

    # Show some examples
    for i, (tokenized_text, label) in enumerate(processed_train):
        # Convert the tokenized text back to readable text
        readable_text = tokenizer.decode(tokenized_text['input_ids'][0])

        print(f"Example {i+1}:")
        print("Readable Text:", readable_text)
        print("Label:", label)
        print("------")

        # Limit to printing only the first few examples
        if i >= 2:  # Adjust this number to print more or fewer examples
            break

    # Show some examples
    for i, (tokenized_text, label) in enumerate(processed_val):
        # Convert the tokenized text back to readable text
        readable_text = tokenizer.decode(tokenized_text['input_ids'][0])

        print(f"Example {i+1}:")
        print("Readable Text:", readable_text)
        print("Label:", label)
        print("------")

        # Limit to printing only the first few examples
        if i >= 2:  # Adjust this number to print more or fewer examples
            break

    