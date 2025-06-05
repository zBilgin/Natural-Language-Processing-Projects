# =============================================
# 🧠 Word Embedding Görselleştirme - Word2Vec & FastText
# Geliştirici: Google (Word2Vec), Meta (FastText)
# =============================================

# 📦 Gerekli Kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt

# PCA: Principal Component Analysis
# Verideki boyutları azaltarak, önemli bilgileri koruyarak
# yeni bir özellik (bileşen) seti oluşturur.
# Böylece yüksek boyutlu veriler görselleştirilebilir hale gelir.
from sklearn.decomposition import PCA  # Boyut indirgeme (dimension reduction)

from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess


# =============================================
# 🧾 Örnek Cümle Listesi
sentences = [
    "Köpek çok tatlı bir hayvandır.",
    "Köpekler evcil hayvanlardır.",
    "Kediler genellikle bağımsız hareket etmeyi severler.",
    "Köpekler sadık ve dost canlısı hayvanlardır.",
    "Hayvanlar insanlar için iyi arkadaşlardır."
]

# 🧼 Ön Temizleme (Basit tokenizasyon işlemi)
# Gensim'in simple_preprocess fonksiyonu küçük harfe çevirme,
# noktalama temizliği vb. işlemleri yapar.
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# =============================================
# 🧠 Word2Vec Modeli Eğitimi
word2vec_model = Word2Vec(
    sentences=tokenized_sentences,   # Eğitim verisi
    vector_size=50,                  # Kelime gömme (embedding) boyutu
    window=5,                        # Bağlam penceresi (ön ve arkadaki 5 kelime)
    min_count=1,                     # Minimum geçme sayısı (1 kere geçen kelimeler dahil)
    sg=0                             # CBOW (sg=0), sg=1 yaparsan Skip-Gram olur
)

# =============================================
# ⚡ FastText Modeli Eğitimi
fasttext_model = FastText(
    sentences=tokenized_sentences,
    vector_size=50,
    window=5,
    min_count=1,
    sg=0
)

# =============================================
# 📊 PCA ile Word Embedding Görselleştirme Fonksiyonu

def plot_word_embedding(model, title):
    word_vectors = model.wv

    # Normalde kelime sayısı çok olabilir, ilk 1000 taneyle sınırla
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]

    # PCA: 50 boyutlu vektörü 3 boyuta indir
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)

    # 3D Görselleştirme
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2])

    # Kelimeleri grafikte etiketle
    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()


# =============================================
# ▶️ Model görselleştirme çağrıları
plot_word_embedding(word2vec_model, "Word2Vec")
plot_word_embedding(fasttext_model, "FastText")

# =============================================
# 🛠️ Spyder'da Grafiklerin Yeni Pencerede Açılması İçin Ayar:
"""
Spyder'da matplotlib grafiklerinin ayrı pencere (figure) olarak açılması için:

1. Tools (Araçlar) → Preferences (Ayarlar)
2. IPython Console → Graphics
3. Backend: [Automatic] yerine [Qt5 (new window)] seç
4. Apply → OK → Spyder'ı yeniden başlat
"""
