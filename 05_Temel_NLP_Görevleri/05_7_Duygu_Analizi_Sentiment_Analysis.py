"""
Problem tanimi ve veriseti:
    https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv
  
  amazon veri seti icerisinde bulunan yorumlarin positive mi yoksa negative mi oldugunu siniflandirmak
  binary classification problemi
"""

# import  libraries
import pandas as pd
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# bu bizim nltk kütüphanesinde bulunan sentiment-duygu
# analiz yapmamızı gerçekleştirecek-sağlayacak olan kütüphanemiz

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

#sentiment analizi için gerekli veri
nltk.download("vader_lexicon")

nltk.download("stopwords")

nltk.download("punkt")

nltk.download("wordnet")

nltk.download("omw-1.4")

# veri seti yükle
df = pd.read_csv("../Datasets/duygu_analizi_amazon_veri_seti.csv")

# lemmatizer tanımlama
lemmatizer = WordNetLemmatizer()

# text cleaning ve preprocessing
def clean_preprocess_data(text):
   
    # tokenize
    tokens = word_tokenize(text.lower())
   
    # stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # join words
    processed_text = " ".join(lemmatized_tokens)

    return processed_text    

df['reviewText2'] = df['reviewText'].apply(clean_preprocess_data)


# sentiment analysisis (nltk)
analyzer = SentimentIntensityAnalyzer()

def get_sentiments(text):
    
    score = analyzer.polarity_scores(text)
    
    sentiment = 1 if score["pos"] > 0 else 0
    
    return sentiment

df["sentiment"] = df["reviewText2"].apply(get_sentiments)

# evuluation - test 
# sentiment analaysis yapmış olduğumuz modelin değerlendirmesi
# ne kadar dogru tahmin gerceklestirdik

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(df["Positive"], df["sentiment"])
print(f"Confusion Matrix: {cm}")

"""
Confusion Matrix:
        0      1
 0- [[ 1131  3636]
 1- [  576 14657]]

"""

cr = classification_report(df["Positive"], df["sentiment"])
print(f"Classification Report: \n{cr}")

"""
Classification Report: 
              precision    recall  f1-score   support

           0       0.66      0.24      0.35      4767
           1       0.80      0.96      0.87     15233

    accuracy                           0.79     20000
   macro avg       0.73      0.60      0.61     20000
weighted avg       0.77      0.79      0.75     20000

"""




