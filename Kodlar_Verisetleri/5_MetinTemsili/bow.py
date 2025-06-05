# count vectorizer iceriye aktar
from sklearn.feature_extraction.text import CountVectorizer

# veri seti olustur
documents = [
    "kedi bahçede",
    "kedi evde"]

# vectorizer tanimla
vectorizer = CountVectorizer()

# metni sayisal vektorlere cevir
X = vectorizer.fit_transform(documents)


# kelime kumesi olusturma [bahçede, evde, kedi]
feature_names = vectorizer.get_feature_names_out()
print(f"kelime kumesi: {feature_names}")

# vektor temsili
vector_temsili = X.toarray()

print(f"vector_temsili: {vector_temsili}")


"""
kelime kumesi: ['bahçede' 'evde' 'kedi']
vector_temsili: 
    [[1 0 1]
     [0 1 1]]
"""



















