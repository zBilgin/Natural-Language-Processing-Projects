# --- Kütüphaneleri İçe Aktar ---
from surprise import Dataset, KNNBasic, accuracy # Surprise kütüphanesi, tavsiye sistemleri için çeşitli araçlar sunar.
from surprise.model_selection import train_test_split # Veri setini eğitim ve test kümelerine ayırmak için.

# --- Veri Setini Yükle ---
# Surprise kütüphanesinin içinde bulunan "ml-100k" (MovieLens 100K) veri setini yükleriz.
# Bu veri seti genellikle (kullanıcı ID, film ID, puan) formatında filmlere verilen derecelendirmeleri içerir.
data = Dataset.load_builtin("ml-100k") 

# --- Eğitim ve Test Kümelerine Ayırma ---
# Yüklenen veri setini eğitim (%80) ve test (%20) kümelerine rastgele ayırırız.
# trainset: Modelin öğreneceği verileri içerir.
# testset: Modelin performansını değerlendireceğimiz, daha önce görmediği verileri içerir.
trainset, testset = train_test_split(data, test_size = 0.2) 

# --- Makine Öğrenimi Modeli Oluşturma: KNN (K-Nearest Neighbors) ---
# KNNBasic, Surprise kütüphanesindeki temel K-En Yakın Komşu algoritmasıdır.
# Bu algoritma, benzer kullanıcıları (veya öğeleri) bularak tahminler yapar.
model_options = {
    "name": "cosine", # Benzerlik metrik olarak "kosinüs benzerliği" (cosine similarity) kullanılacağını belirtir.
                      # Diğer seçenekler arasında 'msd', 'pearson', 'pearson_baseline' bulunur.
    "user_based": True # 'True' ise kullanıcı tabanlı işbirlikçi filtreleme (user-based collaborative filtering) yapılır.
                        # Yani, benzer kullanıcıların puanlarına bakılarak tahmin yapılır.
                        # 'False' ise öğe tabanlı (item-based) işbirlikçi filtreleme yapılır.
}

# KNNBasic modelini belirtilen seçeneklerle başlat.
model = KNNBasic(sim_options=model_options)

# --- Modeli Eğit ---
# Modeli, eğitim veri seti (trainset) üzerinde eğitir.
# Bu adımda model, kullanıcılar arası (veya öğeler arası) benzerlikleri hesaplar ve dahili olarak bir 'komşu' yapısı oluşturur.
model.fit(trainset)

# --- Modeli Test Et ---
# Eğitilmiş modelin, test veri seti (testset) üzerindeki performansını değerlendirir.
# prediction: Her bir test kaydı için (kullanıcı, öğe, gerçek puan, tahmin edilen puan) içeren bir listedir.
prediction = model.test(testset)

# RMSE (Root Mean Squared Error - Ortalama Karekök Hata) metrikini hesaplar.
# RMSE, tahmin edilen puanlarla gerçek puanlar arasındaki farkın karekök ortalamasını verir.
# Daha düşük RMSE değeri, modelin daha iyi tahminler yaptığını gösterir.
accuracy.rmse(prediction)

# --- Tavsiye Sistemi Fonksiyonu ---
def get_top_n(predictions, n = 10):
    """
    Belirli bir kullanıcı için en iyi n adet tavsiyeyi döndürür.

    Parametreler:
        predictions (list): Modelin test veri seti üzerinde yaptığı tahminlerin listesi.
                            Her tahmin (uid, iid, true_r, est, _ ) formatındadır.
        n (int): Her kullanıcı için kaç adet en iyi öğenin tavsiye edileceği sayısı.

    Dönüş:
        dict: Anahtarları kullanıcı ID'leri olan ve değerleri, o kullanıcı için
              tahmin edilen puana göre sıralanmış (öğe ID, tahmin edilen puan) çiftlerinin
              listeleri olan bir sözlük.
    """
    
    # Her kullanıcı için tahmin edilen puanları saklayacak bir sözlük oluştururuz.
    # Yapı: {kullanici_id: [(urun_id, tahmin_puanı), ...]}
    top_n = {}
    
    # Tahminler listesi üzerinde döngü yaparız.
    # uid: Kullanıcı ID'si
    # iid: Öğe (ürün/film) ID'si
    # true_r: Gerçek puan (ground truth rating)
    # est: Tahmin edilen puan (estimated rating)
    # _: Ek bilgi ( Surprise kütüphanesinden gelen diğer parametreler)
    for uid, iid, true_r, est, _ in predictions:
        # Eğer kullanıcı henüz top_n sözlüğünde yoksa, boş bir liste oluştur.
        if not top_n.get(uid): # top_n.get(uid) ile anahtar varsa değeri döner, yoksa None döner.
            top_n[uid] = []
        # Kullanıcının listesine (öğe ID, tahmin edilen puan) çiftini ekle.
        top_n[uid].append((iid, est))
            
    # Her kullanıcı için öğeleri, tahmin edilen puanlara göre sırala ve ilk n tanesini al.
    for uid, user_ratings in top_n.items():
        # user_ratings listesini, her bir (öğe ID, puan) demetindeki puanlara (x[1]) göre ters sırada (büyükten küçüğe) sıralarız.
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        # Sıralanmış listeden sadece ilk n öğeyi (en yüksek puanlıları) alırız.
        top_n[uid] = user_ratings[:n]

    return top_n

# --- Tavsiye Yap ---
n_recommendations = 5 # Her kullanıcı için kaç tavsiye yapılacağını belirler.
top_n_recommendations = get_top_n(prediction, n_recommendations)

# Belirli bir kullanıcı için tavsiyeleri yazdırma örneği.
user_id_to_recommend = "2" # Tavsiye almak istediğimiz kullanıcının ID'si (string olmalı çünkü ml-100k'da kullanıcı ID'leri stringdir).
print(f"\nKullanıcı {user_id_to_recommend} için en iyi {n_recommendations} tavsiye:")
# Seçilen kullanıcının tavsiye listesi üzerinde döngü yaparak ürün ID'sini ve tahmin edilen puanı yazdırırız.
for item_id, rating in top_n_recommendations[user_id_to_recommend]:
    print(f"Ürün ID: {item_id}, Tahmin Edilen Puan: {rating:.2f}") # Puanı 2 ondalık basamakla biçimlendiririz.