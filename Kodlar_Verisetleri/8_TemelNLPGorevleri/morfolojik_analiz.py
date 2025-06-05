import spacy

nlp = spacy.load("en_core_web_sm")

# incelenecek olan kelime yada kelimeler

word = "I go to schools"

# kelimeyi nlp isleminden gecir
doc = nlp(word)

for token in doc:
    
    print(f"Text: {token.text}")            # kelimenin kendisi
    print(f"Lemma: {token.lemma_}")         # kelimenin kok hali
    print(f"POS: {token.pos_}")             # kelimenin dilbilgisel ozelligi
    print(f"Tag: {token.tag_}")             # kelimenin detayli dilbilgisel ozelligi
    print(f"Dependency: {token.dep_}")      # kelimenin rolu 
    print(f"Shape: {token.shape_}")         # karakter yapisi
    print(f"Is alpha: {token.is_alpha}")    # kelimenin yalnizca alfabetik karakterlerden olusup olusmadigini kontrol eder
    print(f"Is stop: {token.is_stop}")      # kelimenin stop words olup olmadigi
    print(f"Morfoloji: {token.morph}")      # kelimenin morfolojik ozelliklerini verir 
    print(f"Is plural: {'Number=Plur' in token.morph}") # kelimenin cogul olup olmadigi
    print()