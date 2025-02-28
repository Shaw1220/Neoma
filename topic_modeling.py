
from bertopic import BERTopic
import pandas as pd
from gensim.models import Word2Vec
import numpy as np

#import the dataset2(There is no sentiment words)
df = pd.read_csv('/Users/mashaojie/Desktop/Neoma/S4/毕业论文/Database/Dataset2.csv', encoding='utf-8-sig')

#select the data range
df = df
#convert the text to string
df['reviews.text'] = df['reviews.text'].fillna('').astype(str)
df['reviews.title'] = df['reviews.title'].fillna('').astype(str)

##The word2vec model
word2vec_model = Word2Vec(
    sentences=[doc.split() for doc in df['reviews.text']],  
    vector_size=100,
    window=5,
    min_count=5,  
    workers=4
)

#set the base themes for the third models
base_themes = {
    "staff": ["staff", "service", "employee", "helpful", "friendly"],
    "breakfast": ["breakfast", "food", "meal", "dining", "restaurant"],
    "clean": ["clean", "cleanliness", "neat", "tidy", "spotless"],
    "location": ["location", "area", "place", "nearby", "central"]
}


theme_vectors = {}
for theme, keywords in base_themes.items():
    valid_words = [word for word in keywords if word in word2vec_model.wv]
    if valid_words:
        vectors = [word2vec_model.wv[word] for word in valid_words]
        theme_vectors[theme] = np.mean(vectors, axis=0)
    else:
        print(f"{theme} don't have any valid words")

word2vec_model.save("word2vec_model.model")


#word embedding
word2vec_model = Word2Vec.load("word2vec_model.model")

class Word2VecEmbedding:
    def __init__(self, model):
        self.model = model

    def embed(self, documents, verbose=False):
        embeddings = []
        doc_count = len(documents)
        
        for i, doc in enumerate(documents):
            if verbose:
                print(f"处理文档 {i+1}/{doc_count}")
                
            words = doc.split()
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.model.vector_size))
                if verbose:
                    print(f"警告：文档 {i+1} 没有找到任何有效词向量")
                    
        return np.array(embeddings)




###Topic Modeling
# create the embedding model
word2vec_embedding = Word2VecEmbedding(word2vec_model)

from umap import UMAP
from hdbscan import HDBSCAN

#Bertopic Model
topic_model = BERTopic(
    embedding_model=word2vec_embedding.embed,
    umap_model=UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.05,
        metric='cosine'
    ),
    hdbscan_model=HDBSCAN(
        min_cluster_size=200,   
        min_samples=2,        
        cluster_selection_epsilon=0.2,
        cluster_selection_method='leaf',
        prediction_data=True
    ),
    nr_topics=6,
    top_n_words=5
)

# train the model
topics, probs = topic_model.fit_transform(df['reviews.text'])
# get the topic information
topic_sizes = topic_model.get_topic_info()
total_reviews = sum(topic_sizes['Count'])

#save the results into the txt file
df['theme'] = topics
#save the dataset
df.to_csv('Dataset2_theme.csv', index=False, encoding='utf-8-sig')

#print the results into the txt file"first_model_results"
with open('first_model_results.txt', 'w', encoding='utf-8') as f:
    for topic_id, topic in topic_model.get_topics().items():
        topic_size = topic_sizes[topic_sizes['Topic'] == topic_id]['Count'].values[0]
        percentage = (topic_size / total_reviews) * 100
        f.write(f"\nTheme {topic_id} ({topic_size} comments, {percentage:.1f}%)\n")
        for word, weight in topic[:5]:
            f.write(f" {word}: {weight*100:.1f}%\n")

print("Saved as 'first_model_results'")








#the seconde model
mask = np.array(topics) == -1
topic_0_docs = df['reviews.text'].iloc[mask]

subtopic_model = BERTopic(
    embedding_model=word2vec_embedding.embed,
    umap_model=UMAP(
        n_neighbors=20,
        n_components=5,
        min_dist=0.1,
        metric='cosine'
    ),
    hdbscan_model=HDBSCAN(
        min_cluster_size=200,
        min_samples=2,
        cluster_selection_epsilon=0.25,
        cluster_selection_method='leaf',
        prediction_data=True
    ),
    nr_topics='auto',
    top_n_words=5
)

#train the second model
subtopics, subprobs = subtopic_model.fit_transform(topic_0_docs)
#save the results
subtopic_sizes = subtopic_model.get_topic_info()
total_subtopic_docs = sum(subtopic_sizes['Count'])

#print the results into the txt file'second_model_results'
with open('second_model_results.txt', 'w', encoding='utf-8') as f:
    for subtopic_id, subtopic in subtopic_model.get_topics().items():
        topic_size = subtopic_sizes[subtopic_sizes['Topic'] == subtopic_id]['Count'].values[0]
        percentage = (topic_size / total_subtopic_docs) * 100
        f.write(f"\nsecond_theme {subtopic_id} ({topic_size} comments, {percentage:.1f}%)\n")
        for word, weight in subtopic[:5]:
            f.write(f" {word}: {weight*100:.1f}%\n")
print("Saved as 'second_model_results'")
# renew the theme sign of data
df.loc[mask, 'theme'] = [f"-1_{x}" if x != -1 else "-1" for x in subtopics]
df.to_csv('Dataset2_theme.csv', index=False, encoding='utf-8-sig')




## The third model
# The classify function
def classify_document(doc):
    doc_words = doc.lower().split()
    doc_vector = np.mean([word2vec_model.wv[word] for word in doc_words if word in word2vec_model.wv], axis=0)
    
    if not isinstance(doc_vector, np.ndarray):  
        return "else"
    
    #calculate the similarity of the document with each theme
    similarities = {}
    for theme, theme_vector in theme_vectors.items():
        similarity = np.dot(doc_vector, theme_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(theme_vector))
        similarities[theme] = similarity
    return max(similarities.items(), key=lambda x: x[1])[0]

#extract the noise data
submask = np.array(subtopics) == -1
subtopic_1_docs = topic_0_docs.iloc[submask]
#use the classify function to calculate the similarity
document_themes = [classify_document(doc) for doc in subtopic_1_docs]

# get the noise data
level2_minus1_mask = df['theme'].astype(str) == '-1'

# renew the theme sign of data
df.loc[level2_minus1_mask, 'theme'] = [f"-1_{theme}" if theme != "其他" else "-1" for theme in document_themes]

#save the results
df.to_csv('Dataset2_theme.csv', index=False, encoding='utf-8-sig')

#print the results into the txt file"third_model_results"
valid_themes = [theme for theme in document_themes if theme != "其他"]
if valid_themes:
    theme_counts = pd.Series(valid_themes).value_counts()
    
    with open('third_model_results.txt', 'w', encoding='utf-8') as f:
        total_docs = len(valid_themes)
        for theme, count in theme_counts.items():
            percentage = (count / total_docs) * 100
            f.write(f"\n{theme}: {count}comments ({percentage:.1f}%)\n")
else:
    print("No themes found")
print("Saved as 'third_model_results'")


