import nltk 
#natural languange toolkit

nltk.download("punkt")
nltk.download("punkt_tab")
#metni kelime ve cumle bazinda tokenlara ayirabilmek icin gerekli

text = "Hello, World! How are you? Hello, hi ..."

#kelime tokenizasyonu: word_tokenize: Metni kelimelere ayirir
#noktalama isareleri ve bosluklar ayri birrer token olarak elde edilecektir

word_tokens = nltk.word_tokenize(text)



#cumle tokenizasyonu: sent_tokenize: metni cumlelere ayirir.
#her bir cumle birer token olur
sentence_tokens = nltk.sent_tokenize(text)
