"""
Solve Classification problem (Sentiment Analysis in NLP) with RNN (Deep Learning based Language Model)

duygu analizi -> bir cumlenin etiketlenmesi (positive ve negative)
restaurant yorumlari degerlendirme
"""

# import libraries
import numpy as np
import pandas as pd

from gensim.models import Word2Vec # metin temsili

from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

# create dataset
data = {
    "text": [
        "Yemekler harikaydı, her şey taze ve lezzetliydi.",
        "Garson çok ilgisizdi, siparişimi unuttular.",
        "Tatlılar gerçekten çok güzeldi, bayıldım!",
        "Yemekler soğuktu ve tadı hiç hoş değildi.",
        "Atmosfer oldukça keyifliydi, tekrar geleceğim.",
        "Fiyatlar biraz yüksekti ama yemekler güzeldi.",
        "Servis kalitesi çok iyiydi, teşekkürler.",
        "Yemek çok geç geldi, sabrım kalmadı.",
        "Lezzetli bir akşam yemeği deneyimledik.",
        "Bu restoranı asla tavsiye etmem, kötüydü.",
        "Mekan çok hoştu, özellikle dekorasyonu.",
        "Yemekler beklediğimden çok daha kötüydü.",
        "Güzel bir akşam geçirdik, teşekkürler.",
        "Yemekler fazlasıyla tuzlu geldi, hiç beğenmedim.",
        "Kahvaltı muhteşemdi, her şeyi denemek istedim.",
        "Fiyatlar oldukça makuldü, çok memnun kaldım.",
        "Garsonlar çok yardımseverdi, teşekkürler.",
        "Yemekler güzel ama servis biraz yavaştı.",
        "Çocuklar için harika bir yer, çok eğlendiler.",
        "Bir daha asla gitmeyeceğim, kötü bir deneyim yaşadım.",
        "Mekanın atmosferi çok keyifliydi.",
        "Yemeklerin tadı harikaydı, özellikle deniz ürünleri.",
        "Şarap menüsü oldukça zengindi, beğendim.",
        "Yemekler sıcak servis edilmedi, hayal kırıklığıydı.",
        "Burgerleri gerçekten çok lezzetliydi.",
        "Tatlıların fiyatı biraz yüksekti ama lezzetliydi.",
        "Hizmet çok yavaştı ama yemekler fena değildi.",
        "Gerçekten güzel bir akşam yemeği deneyimi yaşadık.",
        "Sushi taze ve lezzetliydi, kesinlikle tavsiye ederim.",
        "Garsonlar çok nazik ve yardımseverdi.",
        "Hizmetin daha iyi olmasını beklerdim.",
        "Kahvaltı menüsü oldukça zengindi, çok beğendim.",
        "Yemekler çok lezzetliydi ama servis biraz yavaştı.",
        "Fiyatlar oldukça makuldü, bu kadar iyi hizmete.",
        "Mekan çok temizdi, bu benim için önemli.",
        "Tatlıların çok şekerli olduğunu düşündüm.",
        "Hizmet yavaş ama mekan güzeldi.",
        "Yemeklerin lezzeti harikaydı ama porsiyonlar küçük.",
        "Kendimi çok özel hissettim, teşekkürler.",
        "Güzel bir akşam yemeği, tekrar geleceğim.",
        "Çalışanlar çok güler yüzlüydü.",
        "Pasta çok güzeldi, özellikle çikolatalı.",
        "Biraz beklemek zorunda kaldık ama değdi.",
        "Sadece fiyatlar biraz yüksekti ama lezzet buna değer.",
        "Mekan oldukça kalabalıktı ama hizmet güzel.",
        "Garsonlar çok nazik ama biraz daha hızlı olabilirdi.",
        "Yemeklerin sunumu gerçekten etkileyiciydi.",
        "Yemekler çok lezzetliydi ama garsonlar nazik değildi.",
        "Çok güzel bir akşam yemeği deneyimi yaşadım.",
        "Pasta siparişi verdim ama çok uzun sürdü."
    ],
    "label": [
        "positive", "negative", "positive", "negative", "positive",
        "positive", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "positive", "positive", "positive", "negative", "negative",
        "positive", "positive", "positive", "negative", "positive",
        "negative", "positive", "positive", "positive", "positive",
        "negative", "positive", "positive", "negative", "negative",
        "negative", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "negative",
        "negative", "positive", "positive", "positive", "negative"

    ]
}

df = pd.DataFrame(data)
# %% metin temizleme ve preprocessing: tokenization, padding, label encoding, train test split

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

# padding process 
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print(X.shape)

# label encoding 
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# %% metin temsili: word embedding: word2vec

sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=50, window = 5, min_count=1) 

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]


# %% modelling: build, train ve test rnn modeli 

# build model
model = Sequential()

# embedding
model.add(Embedding(input_dim = len(word_index) + 1, output_dim = embedding_dim, weights = [embedding_matrix], input_length=maxlen, trainable = False))

# RNN layer
model.add(SimpleRNN(50, return_sequences = False))

# output layer
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])

# train model
model.fit(X_train, y_train, epochs=10, batch_size = 2, validation_data=(X_test, y_test))

# evaluate rnn model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# %% cumle siniflandirma calismasi
def classify_sentence(sentence):
    
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen) 
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"
    
    return label

sentence = "Restaurant çok temizdi ve yemekler çok güzeldi"

result = classify_sentence(sentence)
print(f"Result: {result}")











































