"""
Part Of Speech POS: kelimelerin uygun sozcuk turunu bulma calismasi
HMM 

I(Zamir) am a teacher (isim)

"""

# import libraries
import nltk
from nltk.tag import hmm


# ornek training data tanimla
train_data = [
    [("I","PRP"), ("am", "VBP"), ("a", "DT"), ("teacher", "NN")],
    [("You", "PRP"), ("are", "VBP"), ("a", "DT"), ("student", "NN")]
    ]

# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# yeni bir cumle olustur ve cumlenin icerisinde bulunan her bir sozcugun turunu etiketle

test_sentence = "I am a student".split()
tags = hmm_tagger.tag(test_sentence)
print(f"Yeni Cumle: {tags}")

"""
Yeni Cumle: [('I', 'PRP'), ('am', 'VBP'), ('a', 'DT'), ('student', 'NN')]
"""


test_sentence = "He is a driver".split()
tags = hmm_tagger.tag(test_sentence)
print(f"Yeni Cumle: {tags}")

"""
Yeni Cumle: [('He', 'PRP'), ('is', 'PRP'), ('a', 'PRP'), ('driver', 'PRP')]
"""












