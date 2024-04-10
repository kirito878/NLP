import json
from transformers import pipeline
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Initialize the zero-shot classification pipeline with an appropriate model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

def get_gold_indices(utterance, responses, situations):

    gold_indices = []

    combined = [utterance + " " + responses]
    # Classify each situation's relevance
    for index, situation in enumerate(situations):
        situation = situation.replace("[", "").replace("]", "")
        result = classifier(combined, situation, multi_label=False)
        if result[0]['scores'][0] > 0.3:
            gold_indices.append(index)
    
    return gold_indices

            

if __name__ == "__main__":
    # Load the data
    with open('./dataset/train.json', 'r') as file:
        data = json.load(file)

    # gold_indices = get_gold_indices(data)

    # Process each record in the data
    for i, item in enumerate(data):
        utterance = item['u'].replace("[", "").replace("]", "")
        responses = item['r']
        situations = item['s']
        situations_types = item['s.type']
        label_response = item.get('r.label', None)  # None if not present
        label = item.get('s.gold.index', [])  # None if not present

        print(f'Utterance: {utterance} Responses: {responses}')
        print(f'Label: {label}, label.r: {label_response}')
        
        # Classify each situation's relevance
        for index, situation in enumerate(situations):
            situation = situation.replace("[", "").replace("]", "")
            result = classifier([utterance + " " + responses], situation, multi_label=False)
            is_relevant = result[0]['scores'][0] > 0.1  # You can adjust the threshold as needed
            
            print(f"Index: {index:{3}}, Type: {situations_types[index]:{13}}, Sentence: {situation:{60}}, Relevance: {'1' if is_relevant else '0'}, Label: {'1' if index in label else '0'}, Score: {result[0]['scores'][0]:.4f}")

        if i >= 5:
            break

    

    