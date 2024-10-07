import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from dataset import InputDataset, collate_fn
from main import BertForSeq
from tqdm import tqdm
from dataset import accusation_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, dataloader):
    model.eval()  # 切换到评估模式
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # 使用交叉熵损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():  # 不需要计算梯度
        # 使用 tqdm 为验证过程添加进度条
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss  # 计算损失
            total_loss += loss.item()

            # 计算预测值和准确率
            logits = outputs.logits
            preds = (torch.sigmoid(logits) > 0.5).float()  # 将 logits 转换为概率，并进行阈值判断

            # 统计正确预测
            correct_predictions += (preds == labels).sum().item()  # 统计匹配的数量
            total_samples += labels.size(0) * labels.size(1)  # 样本总数

    avg_loss = total_loss / len(dataloader)  # 计算平均损失
    accuracy = correct_predictions / total_samples  # 计算准确率

    return avg_loss, accuracy

if __name__ == '__main__':
    # 加载本地 MacBERT 模型的 Tokenizer 和配置
    model_path = './my_macbert_base'
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 加载验证集
    val_dataset = InputDataset('./final_all_data/exercise_contest/data_test.json', tokenizer, max_len=512)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 加载训练好的模型
    config = BertConfig.from_pretrained(model_path, num_labels=len(accusation_list))  # 根据你的罪名类别数量设置
    model = BertForSeq.from_pretrained(model_path, config=config)
    model.to(device)

    # 评估模型
    avg_loss, accuracy = evaluate_model(model, val_dataloader)
    print(f'Validation Loss: {avg_loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')
