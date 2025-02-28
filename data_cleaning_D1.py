import pandas as pd
##Import the dataset
def read_reviews():
    file_path = "/Users/mashaojie/Desktop/Neoma/S4/毕业论文/Database/Datafiniti_Hotel_Reviews.csv" # 替换为您的实际文件名
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        #select the columns we need
        reviews_data = df[['reviews.rating', 'reviews.title','reviews.text']]
        return reviews_data
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

reviews_data = read_reviews()

###Data cleaning
##remove the empty rows
reviews_data = reviews_data.dropna(subset=['reviews.text'])
reviews_data = reviews_data.dropna(subset=['reviews.title'])

##Text Truncation
##truncate text to max 512 tokens
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def truncate_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_tokens, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True)

reviews_data['reviews.text'] = reviews_data['reviews.text'].apply(truncate_text)
reviews_data['reviews.title'] = reviews_data['reviews.title'].apply(truncate_text)
##case folding
reviews_data['reviews.text'] = reviews_data['reviews.text'].str.lower()
reviews_data['reviews.title'] = reviews_data['reviews.title'].str.lower()

##remove the punctuation
import re
reviews_data['reviews.text'] = reviews_data['reviews.text'].apply(lambda x: re.sub(r'[^\w\s]|_', '', x))
reviews_data['reviews.title'] = reviews_data['reviews.title'].apply(lambda x: re.sub(r'[^\w\s]|_', '', x))

##stopwords removal
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#download the stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# get the stopwords list
stop_words = set(stopwords.words('english'))
stop_words.add('hotel')  # 添加"hotel"作为停用词
stop_words.add('room')
stop_words.add('stay')
stop_words.add('stayed')
#sentiment words
stop_words.add('good')
stop_words.add('great')
stop_words.add('bad')
stop_words.add('nice')
stop_words.add('loved')
stop_words.add('friendly')
stop_words.add('helpful')  
stop_words.add('definitely')  
stop_words.add('enjoyed')
stop_words.add('comfortable')
#set the function to remove the stopwords
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)
#apply the function to the reviews
reviews_data['reviews.text'] = reviews_data['reviews.text'].apply(remove_stopwords)
reviews_data['reviews.title'] = reviews_data['reviews.title'].apply(remove_stopwords)

##lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# define the lemmatization function
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
# apply the function to the reviews
reviews_data['reviews.text'] = reviews_data['reviews.text'].apply(lemmatize_text)
reviews_data['reviews.title'] = reviews_data['reviews.title'].apply(lemmatize_text)

## #save the dataset2
# # Convert the DataFrame to a string
reviews_data = reviews_data.astype(str)

output_path = "/Users/mashaojie/Desktop/Neoma/S4/毕业论文/Database/Dataset2.csv"
reviews_data.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nThe data has been saved in : {output_path}")