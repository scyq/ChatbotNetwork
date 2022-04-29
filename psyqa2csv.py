import json
import re
import csv
from typing import List
import random

DIVIDE = True  # 是否分验证集和测试集
TRAIN_RATIO = 0.8
IF_SPLIT_DATA = True  # 如果要去除多层回复，把IF_SPLIT_DATA改为True
IF_DATA_AUGMENTATION = False  # 如果要增加训练数据，把问题描述也算进去，把IF_DATA_AUGMENTATION改为True
IF_UNIQUE_KEY = True  # 如果要去除重复的问题，把IF_UNIQUE_KEY改为True，改为False可能会不说人话
'''
    我猜测数据集是从一些论坛的问答数据中抽取的
    因此部分answer中包含了多层QA
    一条answer_text好像有多层楼的回答
    被很多特殊符号分隔开了
    主要是中英文井号
    所以利用正则表达式只取第一层回复
'''
re_splitter = "#|＃|■|※|@|<|>"
train_data = "datasets/chat_corpus_train.csv"
validation_data = "datasets/chat_corpus_validation.csv"
test_data = "datasets/chat_corpus_test.csv"
delimiter = ","


def get_qa_pairs(obj) -> List:
    pairs = []
    questions = []
    # 记得去除delimiter
    questions.append(obj["question"].replace(delimiter, ""))  # 简洁的问题
    if IF_DATA_AUGMENTATION:
        questions.append(obj["description"].replace(delimiter, ""))
    answers = obj["answers"]
    for question in questions:
        for answer in answers:
            answer_text = answer["answer_text"].replace(delimiter, "")
            if (IF_SPLIT_DATA and re.search(re_splitter, answer_text)):
                answer_text = re.split(re_splitter, answer_text)
                # 去除列表中的空字符串
                answer_text = [x.strip() for x in answer_text if len(x) > 0][0]
            qa_pair = [question, answer_text]
            pairs.append(qa_pair)
            if IF_UNIQUE_KEY:  # 只执行一遍
                break
    return pairs


# 下函数仅针对PysQA数据集
def json2csv():
    # 该数据集涉密, 所以不能直接提取
    with open("PsyQA_full.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if DIVIDE:
        pairs = []
        for obj in json_data:
            pairs.append(*get_qa_pairs(obj))
        random.shuffle(pairs)
        # 811
        train_nums = (int)(len(pairs) * TRAIN_RATIO)
        validation_test_nums = (int)((len(pairs) - train_nums) / 2)
        train_pairs = pairs[:train_nums]
        validation_pairs = pairs[train_nums:train_nums + validation_test_nums]
        test_pairs = pairs[train_nums + validation_test_nums:]

        with open(train_data, 'w', encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f, delimiter=delimiter)
            for pair in train_pairs:
                csv_writer.writerow(pair)

        with open(validation_data, 'w', encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f, delimiter=delimiter)
            for pair in validation_pairs:
                csv_writer.writerow(pair)

        with open(test_data, 'w', encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f, delimiter=delimiter)
            for pair in test_pairs:
                csv_writer.writerow(pair)

    else:
        with open(train_data, 'w', encoding='utf-8', newline="") as f:
            csv_writer = csv.writer(f, delimiter=delimiter)
            for obj in json_data:
                pairs = get_qa_pairs(obj)
                for pair in pairs:
                    csv_writer.writerow(pair)

    print("json转csv完成")


if __name__ == "__main__":
    json2csv()