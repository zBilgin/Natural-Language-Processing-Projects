import nltk 
nltk.download("wordnet") # wordnet: lemmatization islemi icin gerekli veri tabani
from nltk.stem import PorterStemmer # stemming icin fonksiyon

# porter stemmer nesnesini olustur
stemmer = PorterStemmer()

words = ["running", "runner", "ran", "runs", "better", "go", "went"]

# kelimelerin stem'lerini buluyoruz, bunu yaparken de porter stemmerin stem() fonksiyonunu kullaniyoruz
stems = [stemmer.stem(w) for w in words]
print(f"Stems: {stems}")

# %% lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["running", "runner", "ran", "runs", "better", "go", "went"]
lemmas = [lemmatizer.lemmatize(w, pos="v") for w in words]
print(f"Lemmas: {lemmas}")



















