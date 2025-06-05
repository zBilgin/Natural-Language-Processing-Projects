# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter

# veri setinin iceriye aktarilmasi
df = pd.read_csv("IMDB Dataset.csv")

# metin verilerini alalim
documents = df["review"]
labels = df["sentiment"] # positive veya negative

# metin temizleme
def clean_text(text):
    
    # buyuk kucuk harf cevrimi
    text = text.lower()
    
    # rakamlari temizleme
    text = re.sub(r"\d+", "", text)
    
    # ozel karakterlerin kaldirilmasi
    text = re.sub(r"[^\w\s]", "", text)
    
    # kisa kelimelerin temizlenmesi
    text = " ".join([word for word in text.split() if len(word) > 2])
    
    return text # temizlenmis text'i return et

# metinleri temizle
cleaned_doc = [clean_text(row) for row in documents]


# %% bow
# vectorizer tanimla
vectorizer = CountVectorizer()

# metin -> sayisal hale getir
X = vectorizer.fit_transform(cleaned_doc[:75])

# kelime kumesi goster
feature_names = vectorizer.get_feature_names_out()

# vektor temsili goster
vektor_temsili2 = X.toarray()
print(f"Vektor temsili: {vektor_temsili2}")

df_bow = pd.DataFrame(vektor_temsili2, columns = feature_names)

# kelime frekanslarini goster
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_counts))

# ilk 5 kelimeyi print ettir
most_common_5_words = Counter(word_freq).most_common(5)
print(f"most_common_5_words: {most_common_5_words}")





















