"""
metin üretimi
lstm train with text data
text data = gpt ile olustur
"""

# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# eğitim verisi olustur gpt ile
texts = [
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "Kitap okumak beni gerçekten mutlu ediyor.",
    "Kahve içmeden güne başlamak zor geliyor.",
    "Akşam yemeğinde pizza yemeyi planlıyorum.",
    "Sinemaya gitmek her zaman keyifli bir aktivite.",
    "Sabah koşusu bana enerji veriyor.",
    "Yeni bir dil öğrenmek beni heyecanlandırıyor.",
    "Hafta sonu arkadaşlarımla buluşacağım.",
    "Doğada vakit geçirmek beni rahatlatıyor.",
    "Müzik dinlemek beni motive ediyor.",
    "Yarın önemli bir toplantım var, çok heyecanlıyım.",
    "Yeni bir kitap aldım, bu hafta sonu okumayı planlıyorum.",
    "Akşam dışarıda yemek yemeyi düşünüyorum.",
    "Bu sabah yoga yaparak güne başladım.",
    "Sıcak bir çay içmek beni her zaman rahatlatır.",
    "Şehirde yeni bir restoran açılmış, gitmek istiyorum.",
    "Uzun zamandır bisiklete binmiyordum, bugün binmeyi planlıyorum.",
    "Bahçede çiçekler açmış, harika görünüyor.",
    "Bu hafta çok yoğundum, biraz dinlenmeye ihtiyacım var.",
    "Yeni bir diziye başladım, oldukça sürükleyici.",
    "Çalışmalarımda daha verimli olmak istiyorum.",
    "Bugün alışveriş yapmam gerekiyor, eksiklerim var.",
    "Sabahları yürüyüş yapmak benim için iyi bir alışkanlık oldu.",
    "Yeni tarifler denemek mutfakta zaman geçirmemi keyifli kılıyor.",
    "Uzun zamandır görmediğim bir arkadaşımla buluşacağım.",
    "Ders çalışırken sessiz bir ortamda olmak bana daha çok odaklanmamı sağlıyor.",
    "Bugün işlerimi erken bitirip biraz dinleneceğim.",
    "Hafta sonu piknik yapmayı düşünüyorum.",
    "Yaz tatili için plan yapmaya başladım.",
    "Bugün kendime biraz zaman ayırıp film izlemeyi düşünüyorum.",
    "Kış aylarını seviyorum, çünkü sıcak çikolata içmeyi çok seviyorum.",
    "Telefonumun şarjı bitmek üzere, hemen şarja takmam lazım.",
    "İşimle ilgili yeni projeler üzerinde çalışıyorum.",
    "Bugün spora gitmek beni gerçekten zorlayacak.",
    "Tiyatroya gitmeyi uzun zamandır planlıyordum, bu hafta gitmeyi düşünüyorum.",
    "Yeni müzikler keşfetmek her zaman ilgimi çekiyor.",
    "Yarın sabah erkenden uyanmam gerekiyor.",
    "Dışarıda çok güzel bir hava var, belki biraz bisiklet sürerim.",
    "Bahçeye birkaç yeni bitki ekmeyi düşünüyorum.",
    "Kütüphaneye gidip yeni kitaplar bakacağım.",
    "Evdeki işlerimi bitirip bir kahve molası vereceğim.",
    "Gelecek hafta için bir tatil planı yapıyorum.",
    "Bugün işyerinde toplantılar arka arkaya sıralanmış.",
    "Yeni bir hobi edinmek istiyorum, belki resim yapmayı deneyebilirim.",
    "Bu hafta spor salonuna düzenli olarak gitmeye karar verdim.",
    "Sabahları meditasyon yapmak beni çok rahatlatıyor.",
    "Ailecek bu akşam film gecesi yapacağız.",
    "Yaz mevsimi yaklaşırken, deniz tatili planları yapıyorum.",
    "Bugün işim erken biterse bir kitap okumayı planlıyorum.",
    "Yeni tarifler denemek mutfakta daha fazla zaman geçirmemi sağlıyor.",
    "Bugün alışveriş merkezine gidip ihtiyaçlarımı tamamlayacağım.",
    "Gelecek hafta arkadaşlarımla bir doğa yürüyüşüne çıkmayı düşünüyoruz.",
    "Yatmadan önce biraz müzik dinlemek rahatlamamı sağlıyor.",
    "Bugün biraz daha geç kalktım ve kendime vakit ayırdım.",
    "Yarın işim erken biterse yeni bir diziye başlayabilirim.",
    "Mutfakta yeni bir tatlı denemek istiyorum.",
    "Bu hafta sonu şehir dışında bir yere gitmeyi planlıyoruz.",
    "Yeni bir fotoğraf makinesi aldım, hafta sonu denemek için sabırsızlanıyorum.",
    "Sabahları güneşin doğuşunu izlemek bana enerji veriyor.",
    "İş yoğunluğu arasında kısa bir mola vermek bana iyi geliyor.",
    "Akşam yemeği için dışarıda bir yer arıyorum.",
    "Bugün havalar biraz soğudu, kalın giyinmem gerek.",
    "Yeni bir film izlemek için sinemaya gitmeyi düşünüyorum.",
    "Bu hafta sonu evde dinlenip enerji toplamayı planlıyorum.",
    "İşimle ilgili bir sunum hazırlamam gerekiyor.",
    "Yaz tatili için sahil kasabasına gitmek istiyorum.",
    "Dışarıda yağmur yağıyor, tam kitap okuma havası.",
    "Sabah kahvaltısında yeni bir şeyler denemeyi seviyorum.",
    "Bugün ofiste biraz yoğun bir gün geçirdim.",
    "Kış ayları yaklaşıyor, dolabımı yenilemem gerek.",
    "Yeni bir spor dalı denemek istiyorum, belki yoga.",
    "Hafta sonu için arkadaşlarımla bir etkinlik planlıyoruz.",
    "Bu akşam evde kendime biraz zaman ayırıp dinleneceğim.",
    "Yeni bir bilgisayar oyunu keşfettim, oldukça eğlenceli.",
    "Sabahları kahve içmeden kendime gelemiyorum.",
    "Yarın sabah erkenden uyanıp yürüyüş yapmayı planlıyorum.",
    "Hafta sonu sahilde vakit geçirmek harika olurdu.",
    "Bugün birkaç yeni tarif denedim, oldukça lezzetli oldu.",
    "Bu hafta işte oldukça yoğun geçiyor.",
    "Yarın için önemli bir randevum var.",
    "Bugün dışarıda yemek yemeyi planlıyorum.",
    "Evde temizlik yapmak için güzel bir gün.",
    "Bu sabah spora gitmek beni zorlayacak gibi görünüyor.",
    "Hafta sonu şehir dışına kısa bir kaçamak yapmayı düşünüyoruz.",
    "Sabahları kahvaltı yapmadan güne başlamam mümkün değil.",
    "Yeni bir hobi edinmek bana iyi gelebilir.",
    "Kitap okumak beni her zaman başka bir dünyaya götürüyor.",
    "Bugün çok verimli bir gün geçirdim.",
    "Yarın yeni bir projeye başlamayı planlıyorum.",
    "Bu hafta sonu ailecek vakit geçirmeyi planlıyoruz.",
    "Yeni bir film izledim, oldukça etkileyiciydi.",
    "Sabahları yürüyüş yapmak bana enerji veriyor.",
    "Dışarıda çok güzel bir hava var, belki biraz yürüyüş yaparım.",
    "Akşam yemeği için bir şeyler hazırlamam gerekiyor.",
    "Yarın arkadaşlarımla bir araya geleceğim, sabırsızlanıyorum.",
    "Bu hafta biraz daha fazla dinlenmeye ihtiyacım var.",
    "Bugün işlerimi erken bitirip biraz kitap okuyacağım.",
    "Yeni bir müzik albümü keşfetmek beni mutlu ediyor.",
    "Doğada vakit geçirmek beni her zaman sakinleştiriyor."
]



# %% metin temizleme ve preprocessing:
# tokenization, padding, label encoding 

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) # metinler üzerinde ki kelime frekansi öğren fit et
total_words = len(tokenizer.word_index) + 1 # toplam kelime sayisi

# n-gram dizileri olustur ve padding uygula
input_sequences = []
for text in texts:
    # metinleri kelime indeksilerine cevir
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # her bir metin icin n-gram dizisi olusturalım
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
# en uzun diziyi bulalım, tüm dizileri aynı uzunluga getirelim
max_sequence_length = max(len(x) for x in input_sequences)

# dizilere pading islemi uygula
# hepsinin ayni uzunlukta olmasını sağla
input_sequences = pad_sequences(input_sequences, maxlen= max_sequence_length, padding="pre")


# X(girdi) ve y(hedef-target degisken)
X = input_sequences[:,:-1]
y = input_sequences[:,-1]

y = tf.keras.utils.to_categorical(y, num_classes = total_words) #one hot encoding


# %% LSTM Modeli olustur, complie, train ve evaluate

model = Sequential()

# embeddign
model.add(Embedding(total_words, 50,  input_length= X.shape[1] ))


# lstm
model.add(LSTM(100, return_sequences = False ))

#output
model.add(Dense(total_words, activation = "softmax"))

# model complie
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# model trainig
model.fit(X,y, epochs = 100, verbose = 1)


# model 309 sınıflı bir sınıflandırma modelinde
# acc 89 oluyor overfit oluyor ama az veri bu normal


# %% Model prediction

def generate_text(seed_text, next_words):
    
    for _ in range(next_words):
        
        # girdi metnini sayisal verielere donustur
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # padding
        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_length-1,
            padding="pre")
        
        # prediction
        predicted_probalities = model.predict(token_list,verbose = 0)

        # en yuksek olasiliga sahip kelimenin indexini bul
        predicted_word_index = np.argmax(predicted_probalities, axis = -1)
        
        # tokenizer ile kelime index inden asil kelimeyi bul
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        
        # tahmin edilen kelimeyi seed_text e ekleyelim
        seed_text = seed_text + " " +  predicted_word
        
    return seed_text  # ✅ Artık döngüden sonra dönecek
    
#seed_text = "Bu hafta sonu"
seed_text = "Yarın"
# 2. parametre keç kelime üreteceği
print(generate_text(seed_text, 5 ))


 













