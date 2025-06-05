import spacy 

nlp = spacy.load("en_core_web_sm")

# incelenecek olan kelime yada kelimeler

#word = "book" 
# çıktı: Text: book Lemma: book POS: PROPN Tag: NNP

#word = "went"
# çıtkı: Text: went Lemma: go POS: VERB Tag: VBD

word = "I go to schools 123 123a"
 

# kelimeyi nlp isleminden gecir
doc = nlp(word)

for token in doc:
    
    # kelimenin kendisi
    print(f"Text: {token.text}")
    
    # kelimenin kök hali
    print(f"Lemma: {token.lemma_}")
    
    # kelimenin türü-dilbilgisel özelliği
    print(f"POS: {token.pos_}")   
    
    # kelimenin detaylı dil bilgisel özelliği
    print(f"Tag: {token.tag_}")
    # burada ki VBD verd past tense anlamına gelir
    # go olsa VB Çıkardı
    
    
    # kelimenin rolü  örn root -> ana
    print(f"Dependecy: {token.dep_}")
    
    # kelimenin karakter sayisi-yapisi
    print(f"Shape: {token.shape_}")
    
    # is alpha dediğimiz kavram kelimenin
    # kelimenin yalnizca alfabetik karakterlerden
    # oluşup oluşmadığını kontrol eder
    print(f"Is alpha: {token.is_alpha}")
    
    # kelimenin stop-words olup olmadığı
    print(f"Is stop: {token.is_stop}")
    
    # kelimenin morfolojik özelliklerini verir
    # case: number: sign: vs...
    print(f"Morfoloji: {token.morph}")
    
    # kelimenin çoğul olup olmadığını 
    # kontrol etmek istediğimizde 
    print(f"Is plural: {'Number=Plur' in token.morph}")
    
    
    # boşluk için
    print()
    
    
    
    
    
    