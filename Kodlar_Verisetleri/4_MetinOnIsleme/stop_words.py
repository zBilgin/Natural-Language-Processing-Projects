import nltk

from nltk.corpus import stopwords 

nltk.download("stopwords") # farkli dillerde en cok kullanilan stop words iceren veri seti

# ingilizce stop words analizi (nltk)
stop_words_eng = set(stopwords.words("english"))

# ornek ingilizce metin
text = "There are some examples of handling stop words from some texts."
text_list = text.split()
# eger word ingilizce stop words listesinde (stop_words_eng) yoksa, 
# bu kelimeyi filtrelenmis listeye ekliyoruz
filtered_words = [word for word in text_list if word.lower() not in stop_words_eng]
print(f"filtered_words: {filtered_words}")

# %% turkce stop words analizi (nltk)
stop_words_tr = set(stopwords.words("turkish"))

# ornek turkce metin
metin = "merhaba arkadaslar çok güzel bir ders işliyoruz. Bu ders faydalı mı"
metin_list = metin.split()

filtered_words_tr = [word for word in metin_list if word.lower() not in stop_words_tr]
print(f"filtered_words_tr: {filtered_words_tr}")
# %% kutuphanesiz stop words cikarimi

# stop word listesi olustur
tr_stopwords = ["için", "bu", "ile", "mu", "mi", "özel"]

# ornek turkce metin
metin = "Bu bir denemedir. Amacımiz bu metinde bulunan özel karakterleri elemek mi acaba?"

filtered_words = [word for word in metin.split() if word.lower() not in tr_stopwords]
filtered_stop_words = set([word.lower() for word in metin.split() if word.lower() in tr_stopwords])

print(f"filtered_words: {filtered_words}")















