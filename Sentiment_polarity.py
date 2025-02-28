import pandas as pd
import nltk
from transformers import pipeline, BertTokenizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#use the dataset1 online reviews, it includes the sentiment words
df = pd.read_csv('/Users/mashaojie/PyCharmMiscProject/Dataset1_theme_keywords.csv', encoding='utf-8-sig')
#Text Trunction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def truncate_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_tokens, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True)

df['reviews.text'] = df['reviews.text'].apply(truncate_text)

#transform to the list format
texts = df['reviews.text'].tolist()

#load the sentiment analysis model
classifier = pipeline(
    "text-classification", 
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    max_length=512,
    truncation=True
)

#Data Range
reviews = texts

#get the sentiment scores
sentiment_scores = [classifier(review)[0] for review in reviews]

# create new column sentiment_score
df['sentiment_score'] = [score['label'].replace(' stars', '').replace(' star', '') for score in sentiment_scores]

# the sentiment_poarity function
def get_polarity(score):
    score = int(score)
    if score >= 4:
        return 'positive'
    elif score == 3:
        return 'neutral'
    else:
        return 'negative'

# create the sentiment_polarity column
df['sentiment_polarity'] = df['sentiment_score'].apply(get_polarity)

# save the results
df.to_csv('Dataset1_SentimentScore.csv', index=False, encoding='utf-8-sig')





#visualization
import matplotlib.pyplot as plt
import numpy as np

# 获取每个主题的情感极性统计
themes = ['Cleanliness', 'Location', 'Service', 'Food & Drinks', 'Else']
polarities = ['positive', 'neutral', 'negative']

# 准备数据
data = []
for theme in themes:
    theme_data = []
    theme_df = df[df['theme'] == theme]
    for polarity in polarities:
        count = len(theme_df[theme_df['sentiment_polarity'] == polarity])
        theme_data.append(count)
    data.append(theme_data)

# 设置柱状图参数
x = np.arange(len(themes))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
rects1 = ax.bar(x - width, [d[0] for d in data], width, label='Positive')
rects2 = ax.bar(x, [d[1] for d in data], width, label='Neutral')
rects3 = ax.bar(x + width, [d[2] for d in data], width, label='Negative')

# 设置图表格式
ax.set_ylabel('Count')
ax.set_title('Sentiment Distribution by Theme')
ax.set_xticks(x)
ax.set_xticklabels(themes)
ax.legend()

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()



