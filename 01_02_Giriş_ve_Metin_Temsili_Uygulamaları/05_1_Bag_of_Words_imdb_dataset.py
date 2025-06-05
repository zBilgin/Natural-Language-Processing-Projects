#import libraies
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter

#  veri setinin iceriye aktarilmasi
df = pd.read_csv("Kodlar_Verisetleri/5_MetinTemsili/IMDB_Dataset.csv")

# metin verilerini alalim
documents = df["review"]
labels = df["sentiment"] #positive veya negative

 
# ön temizlik için fonksiyon tanımlama
def clean_text(text):
    
    #buyuk kucuk harf cevrimi
    text = text.lower()
    
    #rakamlari temizleme
    text = re.sub(r"\d+", "", text)
    
    #ozel karakterlerin temizlenmesi
    text = re.sub(r"[^\w\s]", "", text)
    
    #kisa kelimelerin temizlenmesi i,in gibi ing de anlamsiz kelimeleri
    text = " ".join([word for word in text.split() if len(word) > 2])
    
    return text # temizlenmis text i return et
    
#  metin-veri temizleme 
cleaned_doc = [clean_text(row) for row in documents]


# %% BoW

# vectorizer tanimla
vectorizer = CountVectorizer()

# metin -> sayisal hale getir
#burada 50 bin tane veri oldugu icin yapmak uzun sureceginden
#ilk 75 tanesini aldım
X = vectorizer.fit_transform(cleaned_doc[:75])


# kelime kumesi goster
feature_names = vectorizer.get_feature_names_out()


# vektor temsili göster
vektor_temsili2 = X.toarray()
print(f"Vektor Temsili {vektor_temsili2}")

# vektor temsilini df olarak görelim
df_bow = pd.DataFrame(vektor_temsili2, columns = feature_names)

# kelime frekanslari goster
word_counts = X.sum(axis=0).A1

#bu word counts da index olarak görüyoruz
#hangi kelime karsılıgını görmek icin ise
# zipliyoruz
word_freq = dict(zip(feature_names, word_counts))

# Ama dikkat burada stop-words ler çıkmadığı icin
# StopWordsler de var 
# Sen sonradan çıkar ödev olsun

# en cok gecen ilk 5 kelimeyi ve sayisini yazdiralim
most_common_5_words = Counter(word_freq).most_common(5)
print(f"Most Common top 5 words: {most_common_5_words}")




