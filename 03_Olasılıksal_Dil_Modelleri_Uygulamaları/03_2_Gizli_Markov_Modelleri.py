"""
Part Of Speech POS: kelimelerin uygun sozcuk 
turunu bulma calismasi
I(Zamir) am a teacher (isim)

"""

# import libraries
import nltk 
from nltk.tag import hmm

# ornek trainig data tanimla
# burada kelime türlerini de yazmamız lazım
# ingilizcde de ı prp zamir fiil verb vbp
train_data = [
    [("I", "PRP"), ("am","VBP"), ("a","DT"), ("teacher","NN")],
    [("You", "PRP"), ("are","VBP"), ("a","DT"), ("student","NN")],
    ]


# train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# yeni bir cumle oluştur ve 
# cumlenin icerisinde bulunan her bir 
# sozcugun turunu etiketle

test_sentence = "I am a student".split()

tags = hmm_tagger.tag(test_sentence)

print(f"Yeni Cumle: {tags}")

"""
Yeni Cumle: [('I', 'PRP'), ('am', 'VBP'), ('a', 'DT'), ('student', 'NN')]
"""

test_sentence2 = "He is a driver".split()

tags2 = hmm_tagger.tag(test_sentence2)

print(f"Yeni Cumle2: {tags2}")

"""
Yeni Cumle2: [('He', 'PRP'), ('is', 'PRP'), ('a', 'PRP'), ('driver', 'PRP')]
3 ünü yanlış bildi bundan buyuk veriye ihtiyacımız var.
"""

