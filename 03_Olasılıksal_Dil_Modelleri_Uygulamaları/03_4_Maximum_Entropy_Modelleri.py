"""
classsification problems: 
duygu analizi -> olumlu veya olumsuz
olarak siniflandirma
"""
# ==============================================
# 🤖 Maximum Entropy Sınıflandırıcı (Duygu Analizi)
# ==============================================

# 📦 Gerekli Kütüphaneler
from nltk.classify import MaxentClassifier

# ==============================================
# 🧪 Eğitim Verisi (Manual Feature Set)
# veri seti tanimlama
"""
Her örnek bir tuple'dan oluşur:
({özellik: True/False}, etiket)

Özellikler: belirli duygu kelimelerinin varlığı
Etiketler: "positive" veya "negative"
"""
train_data = [
    ({"love": True, "amazing": True, "happy": True, "terrible": False}, "positive"),
    ({"hate": True, "terrible": True}, "negative"),
    ({"joy": True, "happy": True, "hate": False}, "positive"),
    ({"sad": True, "depressed": True, "love": False}, "negative")
]

# ==============================================
# 🧠 MaxEnt Modelinin Eğitimi
# train maximum entropy classifier 
# max_iter: maksimum iterasyon sayısı (daha fazlası daha iyi öğrenme olabilir)
classifier = MaxentClassifier.train(train_data, max_iter=10)

# ==============================================
# 📝 Test Cümlesi ve Özellik Çıkarımı

# cumle ile test
#test_sentence = "ı love like this movie"
test_sentence = "I hate this movie and it was terrible"

# Belirlediğimiz kelimelerin test cümlesinde geçip geçmediğini kontrol et
# Bu, binary özellik vektörü üretir (True/False)
target_words = ["love", "amazing", "terrible", "happy", "joy", "depressed", "sad", "hate"]

# Özellik vektörünü oluştur
features = {word: (word in test_sentence.lower().split()) for word in target_words}

# ==============================================
# ✅ Sınıflandırma (Tahmin)

label = classifier.classify(features)
print(f"🔍 Tahmin Edilen Duygu: {label}")
