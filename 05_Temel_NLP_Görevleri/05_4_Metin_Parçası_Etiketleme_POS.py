import spacy


nlp = spacy.load("en_core_web_sm")

sentence1 = "What is the weather today or tomorrow"
doc1 = nlp(sentence1)

for token in doc1:
    print(token.text, token.pos_)
    
"""
What PRON
is AUX
the DET
weather NOUN
today NOUN
or CCONJ
tomorrow NOUN
"""    