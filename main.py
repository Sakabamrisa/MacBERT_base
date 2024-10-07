# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

# BertForSequence 多标签分类任务模型定义
class BertForSeq(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSeq, self).__init__(config)
        self.num_labels = config.num_labels  # 类别数目
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取 BERT 的输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # 分类层输出 logits
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()  # 使用多标签的损失函数
            loss = loss_fct(logits, labels.float())  # labels 转换为浮点类型

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 预测函数：输入 fact，输出预测的 accusation
def predict_accusation(fact, model, tokenizer, accusation_list, max_len=512, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将 fact 转换为模型输入
    encoding = tokenizer.encode_plus(
        fact,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        probs = torch.sigmoid(logits)  # 将 logits 转换为概率

    probs = probs.cpu().numpy()[0]  # 转换为 numpy 格式
    predicted_labels = (probs > threshold).astype(int)  # 根据阈值判断是否为 1（存在该罪名）

    # 获取预测的罪名
    predicted_accusations = [accusation_list[i] for i, label in enumerate(predicted_labels) if label == 1]

    return predicted_accusations


if __name__ == '__main__':
    # 加载本地 MacBERT 模型的 Tokenizer 和配置
    model_path = './my_macbert_base'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)   # 加载配置文件
    accusation_list = config.accusation_list

    # 加载微调后的 MacBERT 模型
    model = BertForSeq.from_pretrained(model_path, config=config)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 输入的 fact 示例
    fact = "公诉机关指控,2015年11月10日晚9时许，被告人李某的妹妹李某某与被害人华某某在桦川县悦来镇石锅烤肉吃饭时发生口角，华某某殴打李某某被他人拉开。后李某某打电话将此事告知李某。李某便开车接上李某某在悦来镇“0454饮吧”找到华某某并质问其因何殴打李某某，之后二人厮打在一起。李某用拳头、巴掌连续击打华某某脸部，致华受伤住院治疗。经桦川县公安局司法鉴定，华某某所受伤为轻伤二级。"

    # 预测罪名
    predicted_accusations = predict_accusation(fact, model, tokenizer, accusation_list)

    # 输出预测结果
    print(f"Fact: {fact}")
    print(f"Predicted Accusations: {predicted_accusations}")
