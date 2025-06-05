import nltk 
from nltk.wsd import lesk 
# lesk algoritması bir kelimenin anlamını
# belirtlemek için bağlamını-çevresini kullanarak
# sözlükte karşılaştıran bir algoritma


# gerekli nltk paketleri indir
nltk.download("wordnet")
nltk.download("own-1.4")
nltk.download("punkt")
nltk.download("punkt_tab")

# ilk cumle
s1 = " I go to the bank to deposit money"
w1 = "bank"

sense1 = lesk(nltk.word_tokenize(s1),w1)
print(f"Cumle: {s1}")
print(f"Word: {w1}")
print(f"Sense: {sense1.definition()}")

"""
Cumle:  I go to the bank to deposit money
Word: bank
Sense: a container (usually with a slot in the top) for keeping money at home

Paranın saklanması gereken bir konteynır olarak algıladı bir nevi
"""


s2 = "The river bank is flooded after the heavy rain"
w2 = "bank"

sense2 = lesk(nltk.word_tokenize(s2),w2)
print(f"Cumle2: {s2}")
print(f"Word2: {w2}")
print(f"Sense2: {sense2.definition()}")
"""
Cumle2: The river bank is flooded after the heavy rain
Word2: bank
Sense2: a slope in the turn of a road or track; the outside is higher than the inside in order to reduce the effects of centrifugal force
"""




