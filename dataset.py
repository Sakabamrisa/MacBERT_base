from torch.utils.data import Dataset
import torch
import torch.nn.utils.rnn as rnn_utils
import json

accusation_list=[]

# 根据提供的json数据文件，自动生成所有罪名的列表
def generate_accusation_list(data_file):
    accusations = set()  # 使用集合保证罪名唯一性
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            case = json.loads(line)
            for accusation in case['meta']['accusation']:
                accusations.add(accusation)
    return list(accusations)

# 数据集类
class InputDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len):
        self.data = self.load_data(data_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = [self.convert_to_multihot(item['meta']['accusation']) for item in self.data]

    # 从json文件中读取数据
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    # 将罪名列表转换为多热编码向量
    def convert_to_multihot(self, accusations):
        label_vector = [0] * len(accusation_list)
        for accusation in accusations:
            if accusation in accusation_list:
                label_vector[accusation_list.index(accusation)] = 1
        return label_vector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fact = str(self.data[index]['fact'])

        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            fact,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding['token_type_ids'].squeeze(0)
        label_vector = torch.tensor(self.labels[index], dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": label_vector
        }


# 对齐张量，防止维度不统一
def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = rnn_utils.pad_sequence([item['attention_mask'] for item in batch], batch_first=True,
                                            padding_value=0)
    token_type_ids = rnn_utils.pad_sequence([item['token_type_ids'] for item in batch], batch_first=True,
                                            padding_value=0)
    labels = torch.stack([item['labels'] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels
    }

accusation_list = generate_accusation_list('./final_all_data/exercise_contest/data_trainsample.json')
# print(accusation_list)