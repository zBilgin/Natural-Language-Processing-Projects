# import libraries

# metin temsilinde skleanr count vectorizer 
# kullanabiliriz ama farklılık olması açısından 
# farklı kütüphanelerden de ögrenmek icin nltk den
# kullacağız

import nltk
from nltk.util import ngrams # n gram modeli olusturmak icin
from nltk.tokenize import word_tokenize # tokenization

from collections import Counter # acıklamasi

# ornek veri seti olustur
corpus = [
    "I love apple",
    "I love him",
    "I love NLP",
    "You love me",
    "He loves apple",
    "They love apple",
    "I love you and you love me"
    ]


"""
# problem tanimi
    Dil modeli yapmak istiyoruz
    amac 1 kelimeden sonra gelecek kelimeyi
    tahmin etmek: metin uretmek/olusturmak
    bunun icin n-gram dil modeli kullanacağız
    veri seti icin burada tanımladığımız veri seti
    gpt vs de kullanabiliriz ama n-gram alternatifini 
    de inceliyoruz
    
    ex: I...(love)...(apple)

"""
# verileri token haline getir
tokens = [word_tokenize(sentence.lower()) for sentence in corpus]

# bigram 2li kelime grubu olustur
bigrams = []
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list, 2)))

bigrams_freq = Counter(bigrams)




# trigram 3lü kelime grubu olustur
trigrams = []
for token_list in tokens:
    trigrams.extend(list(ngrams(token_list, 3)))

trigrams_freq = Counter(trigrams)



# model testing

# "I Iove” bigram' indan sonra "you" veya "apple” kelimelerinin gelme olasilikla ini hesaplayalim

bigram = ("i", "love") # hedef bigram

# "i love you" olma olasılığı
prob_you = trigrams_freq[("i", "love", "you")]/bigrams_freq[bigram]

print(f"You kelimesinin olma olasılığı: {prob_you}")

# "i love apple" olma olasiliği

prob_apple = trigrams_freq[("i", "love","apple")]/bigrams_freq[bigram]
print(f"Apple kelimesinin olma olasılığı: {prob_apple}")

