import json
import os
import re
import csv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random


def remove(data):
    return re.sub(r'[^\u4e00-\u9fff0-9A-Za-z]', "", data)


def choice_fun(choices_json, dict_tag):
    choices = ','.join([f"\"({dict_tag[i]}) {choice}\"" for i, choice in enumerate(
        choices_json) if i in dict_tag])
    return choices


def preprocess(data, tokenizer, device="cpu"):
    preprocess_data = []
    dict_tag = {0: "A", 1: "B", 2: "C", 3: "D"}
    for item in tqdm(data):
        for item_question in item['questions']:
            paragraph = item['paragraph']
            question = item_question['question']
            choices = choice_fun(item_question['choice'], dict_tag)
            label_choices = []
            for choice in item_question['choice']:
                tmp = remove(choice)
                label_choices.append(tmp)
            id = item_question['id']
            start_prompt = '''
            [|Human|]: 
            舉例1: 段落: "生活中，谁都会遇到被拒绝的情况，我们应该怎样正确对待这种情况呢?首先是要有良好的心理，不管是什么原因被拒绝，自己都不要因此失去自信。因为人人都有可能被拒绝，而且被拒绝的原因有很多，不一定是因为你自己的原因。比如，你邀请别人一起吃饭，可是那人已经跟别的人约好了，只好拒绝了你。这是因为你选择的时间不合适，而不是因为你个人有什么问题。\n其次要弄明白别人拒绝你的理由。比如是他误会了你，或者是你邀请的方式不太好，或者是你的能力不够等等。你知道了这些原因以后就要想办法去改变，那样你就有了下一次成功的机会。所以，有时候被拒绝是一件好事，它可以帮助我们更好地思考一些处理问题的方法。有人说过“被拒绝20次，总会有一次被接受”，很多人就是在不断被拒绝中，学会了正确的处理方法，并且获得最后的成功的。"
            問題: "我们应该怎样对待被拒绝?",
            選項: "(A) 被拒绝之后就失去信心","(B) 被拒绝之后就不再努力","(C) 被拒绝之后不要灰心","(D) 被拒绝之后不问原因",
            [|AI|]: "(C) 被拒绝之后不要灰心"。
            [|Human|]: 
            舉例2: 段落: "三年前，林知善怀着对中医的好奇和喜爱，从韩国大田飞来上海，成为上海中医药大学中医专业的一名本科生。今年暑假，她没有回家休息，而是在上海的医院，寻找实习的机会。\n来中国以前，林知善已经获得了韩国大学中文系的本科学历，真正让她下定决心学中医的是一次看病的经历。“我的胃一直不太好，看了很多西医都不见好转，有一次看中医，我回家吃了大夫开的几服药以后，病情就大有好转，第二次看病时，医生又给我开了饮食处方和运动建议，没过多久胃病就好了。”林知善的中文不算流利，但是，不影Ⅱ向交流。“我过去曾经在电视里看到过中国大夫为病人看病，不用开刀、流血，只用小小的针就能把病人医好了，当时就感到很神奇，甚至觉得他们很帅，治好胃病让我更有体会了，我决定来中国寻找中医治疗的方法。”\n汉语，特别古汉语是外国学生们感觉最难的。“酸”“麻”“闷”等这些词在英语中很难表达，而中医却很讲究望、闻、问、切。\n开处方，这些外国学生都感觉很难。刚开始，学生连最常见的药材和药名都对不上号，但是凭着对中医文化的热爱和老师们的耐心帮助，他们克服了语言障碍。林知善熟练地背诵了“四物汤”“四君子汤”“十全大补汤”的药材和功效。“中医很厉害的一个办法就是‘加减法’，比如生A病的人同时有B病，在处方里添加或减少哪些药材才算对呢?这就是老师高明的地方，他们会活用，我们的水平还不够。”林知善说自己现在最拿手的就是针灸，每次回韩国总要在父母面前表演一下。",
            問題: "三年前，林知善为什么开始学中医?",
            選項: "(A) 暑假不想回家探亲","(B) 想成为一名本科生","(C) 喜欢上海中医药大学","(D) 对中医的好奇和喜爱",
            [|AI|]: "(D) 对中医的好奇和喜爱"。
            [|Human|]: 
            舉例3: 段落: "许多动物的某些器官感觉特别灵敏，它们能比人类提前知道一些灾害事件的发生，例如，海洋中的水母能预报风暴，老鼠能事先躲避矿井崩塌或有害气体，等等。地震往往能使一些动物的某些感觉器官受到刺激而发生异常反应。如一个地区的重力发生变异，某些动物可能通过它们的平衡器官感觉到；一种振动异常，某些动物的听觉器官也许能够察觉出来。地震前地下岩层早已在逐日缓慢活动，而断层面之间又具有强大的摩擦力。这种摩擦力会产生一种低于人的听觉所能感觉到的低频声波。人对每秒20次以上的声波才能感觉到，而动物则不然。那些感觉十分灵敏的动物，在感触到这种低声波时，便会惊恐万状，以至出现冬蛇出洞、鱼跃水面等异常现象。",
            問題: "动物的器官感觉与人的相比有什么不同?"
            "選項": "(A) 没有人的灵敏","(B) 和人的差不多","(C) 和人的一样好","(D) 比人的灵敏"
            [|AI|]: "(D) 对中医的好奇和喜爱"。
            首先閱讀段落，接著根據問題，從選項中進行選擇，一定要選出一個，只要回答選項，不需要解釋，不要任何多餘的話。
            '''
            paragraph = " 段落: " + paragraph
            question = " 問題: " + question
            choices = " 選項: " + choices
            prompt = start_prompt + paragraph + question + choices
            formatted_text = "[|Human|]: " + \
                prompt + " end_of_question"+"[|AI|]: "
            tokenized_text = tokenizer.encode_plus(
                formatted_text, return_tensors='pt', max_length=1680, truncation=True).to(device)
            data = (tokenized_text, label_choices, id)
            preprocess_data.append(data)
    return preprocess_data


with open(os.path.join('dataset', 'test.json'), 'r') as file:
    test_data = json.load(file)

# hyperparameter
trust_remote_code = True
torch_dtype = torch.bfloat16
device_map = "cuda:0"
model_name = "vivo-ai/BlueLM-7B-Chat"
use_fast = False
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=trust_remote_code, use_fast=use_fast)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
preprocess_test_data = preprocess(test_data, tokenizer, device_map)
count = 0
preds = []
model = model.eval()
with torch.no_grad():
    for item in tqdm(preprocess_test_data):
        tokenized_text, choices, id = item
        pred = model.generate(
            **tokenized_text, max_new_tokens=1680, repetition_penalty=1.1)
        pred = pred.cpu()[0]
        decode_text = tokenizer.decode(pred, skip_special_tokens=True)
        return_answer = decode_text.split(" ")
        flag = False
        answer = 999

        for i in return_answer:
            if "end_of_question" in i:
                flag = True
                continue
            if flag == True:
                if remove(i) == "A" or "(A)" in i or "A." in i or "（A）" in i or ("A" in i and "AI" not in i):
                    answer = 0
                    break
                elif remove(i) == "B" or "(B)" in i or "B." in i or "（B）" in i or "B" in i:
                    answer = 1
                    break
                elif remove(i) == "C" or "(C)" in i or "C." in i or "（C）" in i or "C" in i:
                    answer = 2
                    break
                elif remove(i) == "D" or "(D)" in i or "D." in i or "（D）" in i or "D" in i:
                    answer = 3
                    break
        if answer == 999:
            count += 1
            # print(return_answer)
            return_answer = remove(return_answer[-1])
            for index, choice in enumerate(choices):
                if choice in return_answer:
                    answer = index
                    break
                if index == len(choices) - 1:
                    answer = random.randint(0, index)
        preds.append([id, answer])
        # break
print(count)
with open('submission.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'answer'])
    for id, answer in preds:
        writer.writerow([id, answer])
