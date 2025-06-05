import nltk
from nltk.wsd import lesk

# gerekli nltk paketlerini indir
nltk.download("wordnet")
nltk.download("own-1.4")
nltk.download("punkt")

# ilk cumle
s1 = " I go to the bank to deposit money"
w1 = "bank"

sense1 = lesk(nltk.word_tokenize(s1), w1)
print(f"Cumle: {s1}")
print(f"Word: {w1}")
print(f"Sense: {sense1.definition()}")

"""
Cumle:  I go to the bank to deposit money
Word: bank
Sense: a container (usually with a slot in the top) for keeping money at home
"""

s2 = "The river bank is flooded after the heavy rain"
w2 = "bank"
sense2 = lesk(nltk.word_tokenize(s2), w2)

print(f"Cumle: {s2}")
print(f"Word: {w2}")
print(f"Sense: {sense2.definition()}")

"""
Cumle: The river bank is flooded after the heavy rain
Word: bank
Sense: a slope in the turn of a road or track; the outside is higher than the inside in order
"""



















