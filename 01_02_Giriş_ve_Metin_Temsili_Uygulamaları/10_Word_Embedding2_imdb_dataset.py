
# =============================================
# 🧠 Word2Vec + KMeans + PCA: Kelime Kümeleme ve 2D Görselleştirme (IMDB Dataset)
# =============================================

# 📦 Gerekli Kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# =============================================
# 📥 Veri Seti Yükleme

# IMDB film yorumlarını içeren veri setini oku
# "review" sütunu pozitif/negatif film yorumları içeriyor
df = pd.read_csv("Datasets/IMDB_Dataset.csv")
documents = df["review"]

# =============================================
# 🧼 Veri Temizleme Fonksiyonu

def clean_text(text):
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r"\d+", "", text)  # Sayıları kaldır
    text = re.sub(r"[^\w\s]", "", text)  # Noktalama ve özel karakterleri kaldır
    text = " ".join([word for word in text.split() if len(word) > 2])  # Kısa anlamsız kelimeleri çıkar
    #text = simple_preprocess(text)
    return text

# Örnek test
# clean_text("ASDASD 1551 %& I merhabA")

# Tüm belgeleri temizle
cleaned_documents = [clean_text(doc) for doc in documents]

# =============================================
# 🔤 Tokenizasyon İşlemi

# Gensim'in simple_preprocess fonksiyonu kullanılarak her cümleyi kelimelere ayır
# Bu adım clean_text() içine de gömülebilirdi ama eğitimsel amaçla ayrı tutuldu
#tokenize etmeyi clean_text fonk icinde de yapabilirdik yukarıda ki 
#fonk icinde ki yorum satırı gibi
# neden boyle yapmadık veri temizleme ve prpeocesing tokenization
# islemini ayrı yapmak görmek icin

tokenized_documents = [simple_preprocess(doc) for doc in cleaned_documents]

# =============================================
# 🧠 Word2Vec Modeli Eğitimi

model = Word2Vec(
    sentences=tokenized_documents,
    vector_size=50,  # Her kelimeyi 50 boyutlu vektörle temsil et
    window=5,        # Bağlam penceresi: sağ-sol 5 kelime
    min_count=1,     # En az 1 kez geçen kelimeleri al
    sg=0             # CBOW (0), sg=1 olursa Skip-Gram
)

#burada vectorsize önemli bir parametre
# vector size 2 alırsak pca boyut indirgemsi gerek kalmaz
# zaten bize 2 li boyutta bir ayrım verir 

word_vectors = model.wv  # Eğitim sonrası kelime vektörlerine erişim

# İlk 500 kelimeyi al (görselleştirme için yeterli)
words = list(word_vectors.index_to_key)[:500]
vectors = [word_vectors[word] for word in words]  # Vektör değerleri

# =============================================
# 🔗 KMeans ile Kümeleme

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(vectors)
clusters = kmeans.labels_  # Her kelime için küme etiketi (0 veya 1)

# =============================================
# 📉 PCA ile Boyut İndirgeme (50 → 2)

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)  # 2 boyutlu hale getir

# =============================================
# 📊 2D Görselleştirme

plt.figure(figsize=(12, 8))
plt.scatter(
    reduced_vectors[:, 0], reduced_vectors[:, 1],
    c=clusters,
    cmap="viridis",
    s=30
)

# Kümelerin merkezlerini işaretle
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0], centers[:, 1],
    c="red",
    marker="x",
    s=150,
    label="Cluster Centers"
)
plt.legend()

# Her noktanın üstüne kelime etiketlerini ekle
for i, word in enumerate(words):
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], word, fontsize=7)

plt.title("Word2Vec + KMeans Clustering (2D PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# =============================================
# 📝 Notlar
# - Aslında Word2Vec vektörleri 50 boyutlu, ancak biz PCA ile 2 boyuta indiriyoruz.
# - Kümeleme 50 boyutta yapılır, ardından bu sonuçlar 2D'ye indirgenir.
# - Bu nedenle iç içe geçmiş gibi görünse de, yüksek boyutlu uzayda farklı olabilirler.

# ✅ Ödev Önerileri:
# - Stop words (gereksiz kelimeler) temizlenerek daha anlamlı sonuçlar alınabilir.
# - K (küme sayısı) farklı değerlerde denenebilir (örn. K=3, K=5).
# - Word2Vec parametreleri (vector_size, window, sg) değiştirilip etkisi gözlemlenebilir.
