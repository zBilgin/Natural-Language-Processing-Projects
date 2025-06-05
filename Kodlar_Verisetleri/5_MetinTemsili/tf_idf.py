# import libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# ornek belge olustur
documents = [
    "Köpek çok tatlı bir hayvandır",
    "Köpek ve kuşlar çok tatlı hayvanlardır.",
    "Inekler süt üretirler."
    ]

# vektorizer tanimla
tfidf_vectorizer = TfidfVectorizer()

# metinleri sayisal hale cevir
X = tfidf_vectorizer.fit_transform(documents)

# kelime kumesini incele
feature_names = tfidf_vectorizer.get_feature_names_out()

# vektor temsilini incele
vektor_temsili = X.toarray()
print(f"tf-idf: {vektor_temsili}")

df_tfidf = pd.DataFrame(vektor_temsili, columns=feature_names)

# ortalama tf idf degerlerine bakalim
tf_idf = df_tfidf.mean(axis=0)


















