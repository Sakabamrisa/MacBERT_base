import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# 设置随机种子以确保结果可复现
import random
random.seed(42)

# 随机抽取数据并保存
def sample_and_save_data(input_file, output_file, sample_size=30000):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 随机抽取sample_size条数据
    sampled_data = random.sample(data, sample_size)

    # 将抽样的数据写入新的json文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in sampled_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully saved {sample_size} samples to {output_file}.")

#创建data_trainsample.json文件
#sample_and_save_data('./final_all_data/exercise_contest/data_train.json','./final_all_data/exercise_contest/data_trainsample.json')

# 读取数据
data_file = "./final_all_data/exercise_contest/data_trainsample.json"
with open(data_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 转换数据为DataFrame
df = pd.DataFrame(data)

# 句子长度统计
df['sentence_length'] = df['fact'].apply(len)

# 标签分布统计
tags = []
for meta in df['meta']:
    accusations = meta['accusation']
    tags.extend([x for x in accusations if x])  # 确保标签存在

# 创建一个Series来方便统计
tag_series = pd.Series(tags).value_counts()

# 绘制句子长度的直方图
plt.figure(figsize=(10, 6))
sns.histplot(df['sentence_length'], bins=30, kde=True)
plt.title('Sentence Length Distribution')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.show()

# 绘制标签分布的饼图
plt.figure(figsize=(80, 80))
plt.pie(tag_series, labels=tag_series.index, autopct='%1.1f%%', startangle=140)
plt.title('Tag Distribution')
plt.axis('equal')
plt.show()