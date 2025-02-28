import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy as np

# load the Dataset1_SentimentScore.csv
df = pd.read_csv('/Users/mashaojie/PyCharmMiscProject/Dataset1_SentimentScore.csv', encoding='utf-8-sig')
# select the `reviews.text` and `keywords` columns
comments = df['reviews.text']
keywords = df['keywords']

# keyword contribution
themes = ['Cleanliness', 'Location', 'Service', 'Food & Drinks', 'Else']

for theme_name in themes:
    # select the comments and keywords for the current theme
    theme_df = df[df['theme'] == theme_name]
    theme_comments = theme_df['reviews.text']
    theme_keywords = theme_df['keywords']

    # keyword counts
    keyword_counts = {}

    # load the trained Word2Vec model 
    model = Word2Vec.load('/Users/mashaojie/PyCharmMiscProject/word2vec_model.model')
    
    # vetorize the text
    def get_text_vector(text):
        words = text.split()
        # word weight
        word_weights = {}
        total_words = len(words)
        
        # word frequency
        word_freq = {}
        for word in words:
            if word in model.wv:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # location weight and frequency weight
        for i, word in enumerate(words):
            if word in model.wv:
                position_weight = 1 - (i / total_words)  #location weight
                freq_weight = word_freq[word] / total_words  # frequency weight
                word_weights[word] = (1 + position_weight) * (1 + freq_weight)  # 综合权重
        
        weighted_vectors = []
        weights = []
        for word, weight in word_weights.items():
            weighted_vectors.append(model.wv[word] * weight)
            weights.append(weight)
        
        if weighted_vectors:
            return np.average(weighted_vectors, axis=0, weights=weights)
        return np.zeros(model.vector_size)
    
    # similiarity calculation
    for comment, keyword_str in zip(theme_comments, theme_keywords):
        #the model has already loaded
        comment_vector = get_text_vector(comment)
        keywords = [kw.strip() for kw in keyword_str.split(',')]
        
        for kw in keywords:
            if kw and kw in model.wv:
                kw_vector = model.wv[kw]
                similarity = np.dot(comment_vector, kw_vector) / (np.linalg.norm(comment_vector) * np.linalg.norm(kw_vector))
                
                # adjust the weight of the keyword according to the similarity
                # similarity range: [-1, 1], convert to [0, 1] for weight calculation
                normalized_similarity = (similarity + 1) / 2  
                # adjust the weight according to the similarity
                # 显著增加权重对比度
                if normalized_similarity > 0.8:
                    weight = 4.0
                elif normalized_similarity > 0.6:
                    weight = 2.0   
                elif normalized_similarity > 0.4:
                    weight = 1.0     
                elif normalized_similarity > 0.2:
                    weight = 0.5    
                else:
                    weight = 0.1    
                
                keyword_counts[kw] = keyword_counts.get(kw, 0) + weight

    # The total match times and percentage of each keyword
    if keyword_counts:
        total_matches = sum(keyword_counts.values())
        percentages = {kw: (count/total_matches * 100) for kw, count in keyword_counts.items()}

        # print the result
        print(f"\n{theme_name}")
        print(f"{len(theme_comments)}comments")
        for kw, percentage in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
            print(f"{kw}: {percentage:.2f}%")
        print("-" * 50)
    else:
        print(f"\n{theme_name}error")





    # 在计算完百分比后添加绘图代码
    if keyword_counts:
        # 创建柱状图，固定图形大小
        plt.figure(figsize=(8, 5))  # 固定图形大小
        keywords = list(percentages.keys())
        values = list(percentages.values())
        
        # 设置更细的柱子宽度和y轴范围
        plt.bar(keywords, values, width=0.2)  # 将柱子宽度减小为0.2
        plt.ylim(0, 60)  # 设置y轴范围为0-60
        
        plt.title(f'{theme_name} ')
        plt.xlabel('keywords')
        plt.ylabel('percentage (%)')
        
        # 在柱子上方显示具体数值
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()



