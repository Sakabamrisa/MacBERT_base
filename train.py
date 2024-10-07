import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import InputDataset, collate_fn
from main import BertForSeq  # 从main.py导入BertForSeq模型
from tqdm import tqdm  # 引入tqdm来显示进度条
from dataset import accusation_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(train_dataset, val_dataset, batch_size, epochs):
    # 加载本地 MacBERT-base 模型配置
    config = BertConfig.from_pretrained('./chinese_macbert_base', num_labels=len(accusation_list))
    # 将 accusation_list 保存到配置文件中
    config.accusation_list = accusation_list  # 自定义字段，将罪名列表保存到配置中
    # 加载本地 MacBERT-base 模型
    model = BertForSeq.from_pretrained('./chinese_macbert_base', config=config)
    model.to(device)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=1e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    # 用于保存 batch 损失和 epoch 损失的列表
    batch_loss_history = []  # 每个 batch 的损失
    epoch_loss_history = []  # 每个 epoch 的平均损失

    # 开始训练
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # 添加tqdm进度条
        print(f"Epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")  # 使用tqdm为训练数据加载器添加进度条

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # 清空之前的梯度
            model.zero_grad()

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss

            # 反向传播并更新参数
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            # 记录每个 batch 的损失
            batch_loss_history.append(loss.item())
            # 更新进度条显示的损失值
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_loss_history.append(avg_train_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}')

        # 评估模型在验证集上的表现
        evaluate_model(model, val_dataloader)
    # 绘制损失变化图
    plot_loss_curves(batch_loss_history, epoch_loss_history)
    # 保存训练好的模型和 tokenizer
    model.save_pretrained('./my_macbert_base')  # 指定保存目录
    tokenizer.save_pretrained('./my_macbert_base')  # 保存 tokenizer
    config.save_pretrained('./my_macbert_base')  # 保存包含 accusation_list 的 config 文件
    print("Model and tokenizer saved to ./my_macbert_base")

def evaluate_model(model, val_dataloader):
    model.eval()
    total_val_loss = 0

    # 添加tqdm进度条，显示验证过程
    progress_bar = tqdm(val_dataloader, desc="Validating")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

            # 更新进度条显示的验证损失
            progress_bar.set_postfix({'val_loss': loss.item()})

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f'Validation Loss: {avg_val_loss}')

# 绘制loss曲线
def plot_loss_curves(batch_loss_history, epoch_loss_history):
    plt.figure(figsize=(14, 6))

    # 画出 loss with batch 的图像
    plt.subplot(1, 2, 1)
    plt.plot(batch_loss_history, label='Loss with Batch', color='blue')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Loss with Batch')
    plt.legend()
    plt.grid(True)

    # 画出 loss with epoch 的图像
    plt.subplot(1, 2, 2)
    plt.plot(epoch_loss_history, label='Loss with Epoch', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss with Epoch')
    plt.legend()
    plt.grid(True)

    # 展示图像
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 加载本地 MacBERT 模型的 Tokenizer
    tokenizer = BertTokenizer.from_pretrained('./chinese_macbert_base')

    # 加载训练集和验证集
    train_dataset = InputDataset('./final_all_data/exercise_contest/data_trainsample.json', tokenizer, max_len=512)
    val_dataset = InputDataset('./final_all_data/exercise_contest/data_valid.json', tokenizer, max_len=512)

    # 开始训练
    train_model(train_dataset, val_dataset, batch_size=4, epochs=3)
