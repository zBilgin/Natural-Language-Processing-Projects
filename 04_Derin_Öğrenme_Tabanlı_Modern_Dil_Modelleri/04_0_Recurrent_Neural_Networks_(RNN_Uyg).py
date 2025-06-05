# =============================================
# 📊 RNN ile Duygu Analizi (Sentiment Analysis)
# Yorum verilerini kullanarak pozitif/negatif sınıflandırma
# =============================================
"""
Problem ve tanımı:
    Duygu analizi -> Sentiment Analysis
    Bir cumlenin etiketlenmesi
    (positive ve negatif)
    
    Yani sınıflandırma
    Solve Classification problem (sentiment Analysis in NLP)
    with RNN (Deep learning based Language Model)

    Restoran müşteri yorumları degerlendirme

"""

# import libraries
import numpy as np
import pandas as pd

from gensim.models import Word2Vec  # metin temsili için

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

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


# %% metin temizleme ve propecessing:
# tokenization, padding, label encoding , train test split


# tokenization
# Tokenizer: kelimelere sayı ataması yapar ve dizilere çevirir
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

# padding process
# padding: bizim cümlelerimiz içerisinde kelime sayılari farklı
# bizim bunları fixlememiz lazım ondan padding kullanıyoruz
# Padding: tüm cümleleri eşit uzunluğa getiriyoruz
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences = sequences, maxlen = maxlen )
print(X.shape)


# Label encoding: pozitif/negatif -> 1/0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
# y de ki 1 ve 0 pozitif egatif karşılık gelir 


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# çok az oldu çünkü veri 50 satır çok az fazla olması lazim normalde


# %% metin temsili : word embedding: word2vec

sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1)
# UYARIIII dataframe text kullanarak aslında cümlelermizi elde ediyoruz
# split ediyoruz sonrasında word index ile eşeltirityoruz
# burada doğrudan wordindex kullanarak da yapabilirdik ama
# df kullanarak tokenlaştırarak word_index ile eşleştirip
# embedding matrisinimizi oluştuyorujz


# burada ne yapıyporuz: burada eğitim sırasında olusturunaş
# word vektörünün kelime modelleri gömme matrisine ekleniyor
# embedding matrisine bu işlem kelimeleri sayısal biçimde 
# temsil etmek için yaoğtımız bir işlem öncelikle burada ji word index
# isimli bir değişken kullanarak her kelimenin sıralı indexi belirlenip
# model de oluğ olmadığı kontrol edilip ekleniyor 
# Girdi katmanı: önceden eğitilmiş embedding matrisi ile
embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1 , embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# %% modelling: build, traing ve test rnn modeli

# build model 
model = Sequential()

# embedding
model.add(Embedding(input_dim= len(word_index)+1, 
                    output_dim= embedding_dim,
                    weights= [embedding_matrix],
                    input_length= maxlen,
                    trainable = False   
                    ))

# RNN layer
model.add(SimpleRNN(50, return_sequences=False))


# output layer
model.add(Dense(1, activation="sigmoid"))


# complie model
model.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])


# train model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test,y_test))


# evaluate rnn model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")



# %% cumle siniflandirma calismasi
# sanki daha öncesinde bir app varmışda 
# birileri yorumları yuklemis bizim
# algortimamızda bu yourumları degerlendirecekmiş gibi
def classify_sentence(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen= maxlen)
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"
    
    return prediction,label

sentence = "Restaurant çok temizdi ve yemekler çok güzeldi, beğendik güzel"

result = classify_sentence(sentence)
print(f"Result: {result}")



"""
    Ödev
Data boyutunu arttır
E-ticaret ürün örneği gibi
epoch sayısı değiştir
Veri boyutunu artır (örn. 500+ yorum)
Stop-words temizliği ve lemmatization ekle
Daha karmaşık modeller (LSTM, Bidirectional RNN, Attention)
Modeli kaydet & Flask ile servis et
"""

