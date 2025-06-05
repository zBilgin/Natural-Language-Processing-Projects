# import library
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# veri seti yukle
df = pd.read_csv("Kodlar_Verisetleri/5_MetinTemsili/sms_spam.csv")

# tf-idf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.text)

# kelime kumesi incele
feature_names = vectorizer.get_feature_names_out()

# cok fazla 1000 adet gibi anlamsız kelime var
# bundan dolayı veri temizleme blogu-aşaması yap
# stopwords lerden kurtul verileri temizle
# temizleme kısmı odev

# her kelimenin ortalama tf-idf degerleri
tfidf_score = X.mean(axis=0).A1 
 

# tf-idf skorlarini iceren df olustur
df_tfidf = pd.DataFrame({"word":feature_names, "tf-idf score":tfidf_score})

# skorları sırala ve sonucları incele
# score göre azalan bir sıralama
df_tfidf_sorted = df_tfidf.sort_values(by="tf-idf score", ascending=False)
print(f"df_tfidf_sorted top 5:\n  {df_tfidf_sorted.head()}")

