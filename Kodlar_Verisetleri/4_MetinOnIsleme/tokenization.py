import nltk # natural language toolkit

nltk.download("punkt") # metni kelime ve cumle bazinda tokenlara ayirabilmek icin gerekli

text = "Hello, World! How are you? Hello, hi ..."

# kelime tokenizasyonu: word_tokenize: metni keliemlere ayirir, 
# noktalama isaretleri ve bosluklar ayri birer token olarak elde edilecektir.
word_tokens = nltk.word_tokenize(text)

# cumle tokenizasyonu: sent_tokenize: metni cumlelere ayirir. her bir cumle birer token olur.
sentence_tokens = nltk.sent_tokenize(text)
