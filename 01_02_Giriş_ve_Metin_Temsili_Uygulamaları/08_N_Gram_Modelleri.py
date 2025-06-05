# import library
from sklearn.feature_extraction.text import CountVectorizer


# ornek metin
documents = [
    "Bu çalışma NGram çalışmasıdır",
    "Bu çalışma doğal dil işleme çalışmasıdır."
    ]

# unigram, bigram, trigram seklinde
# 3 farkli N degerine sahip gram modeli
# N gram range defaul 1 dir ngram_range = (1,1) seklinde kullanılır
vectorizer_unigram = CountVectorizer(ngram_range = (1,1)) 
vectorizer_bigram = CountVectorizer(ngram_range = (2,2)) 
vectorizer_trigram = CountVectorizer(ngram_range = (3,3)) 

# unigram
X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()
print(f"unigram_features:\n {unigram_features}\n")

# bigram
X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()
print(f"bigram_features:\n {bigram_features}\n")
 
# trigram
X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()
print(f"trigram_features:\n {trigram_features}\n")
 
# sonuclarin incelenmesi