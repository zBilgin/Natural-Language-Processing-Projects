import nltk

nltk.download("wordnet")
#wordnet: Lemmitization islemi icin 
#gerekli veri tabani

from nltk.stem import PorterStemmer
#stemming icin fonksiyon

#porter stemmer nesnesi olustur
stemmer = PorterStemmer()

words = ["running", "runner", "ran", "runs", "better", "go", "went"]


# kelimelerin stemlerini yani köklerini buluyoruz
#bunu yaparken de porter stemmer in stem() fonksiyonunu kullanıyoruz

stems = [stemmer.stem(w) for w in words]
# list comprehension bu örnek
#for döngüsü sağlar çıktıyı da list olarak doner

print(f"Stems: {stems}")

# %% Lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ["running", "runner", "ran", "runs", "better", "go", "went"]

# burada kelimelerin tipini belirtmek icin
#örn fiil, isim vb ikinici parametre olarak
# pos="v" diyorum verb fiil olarak algılıyor
# POS = "v" yani fiil olarak tanı (daha doğru sonuçlar verir)
lemmas = [lemmatizer.lemmatize(w) for w in words]
lemmas2 = [lemmatizer.lemmatize(w, pos="v") for w in words]

print(f"Lemmas: {lemmas}")
print(f"Lemmas2: {lemmas2}")

