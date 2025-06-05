#count vectorizer iceriye aktar
from sklearn.feature_extraction.text import CountVectorizer


#veri seti olustur
documents = [
    "kedi bahçede",
    "kedi evde"
    ]

# vectorizer tanımla
# vectorizer : Metin temsilini yapacak olan degisken
vectorizer = CountVectorizer()


# metni sayısal vektorlere cevir
X = vectorizer.fit_transform(documents)


#kelime kümesi olusturma [bahçede,evde, kedi ]
feature_names = vectorizer.get_feature_names_out()

#vektor temsili
vector_temsili = X.toarray()

print("Kelime Kümesi:", feature_names)
print("BoW Vektör Temsili:\n",vector_temsili)