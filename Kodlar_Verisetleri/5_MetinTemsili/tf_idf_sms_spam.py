# import library
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# veri seti yukle
df = pd.read_csv("sms_spam.csv")

# veri temizleme hw

# tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.text)

# kelime kumesini incele
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1 # her kelimenin ortalama tf-idf degerleri

# tfidf skorlarini iceren bir df olustur
df_tfidf = pd.DataFrame({"word":feature_names, "tfidf_score": tfidf_score})

# skorlari sirala ve sonuclari incele
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))