# --- Kütüphaneleri İçe Aktar ---
import numpy as np # Sayısal işlemler ve diziler için

# Sinir ağı kullanacağımız için Keras kütüphanesini kullanacağız.
# Keras, TensorFlow üzerine inşa edildiği için TensorFlow kütüphanesi üzerinden çağrılır.
from tensorflow.keras.models import Model 

# Modele ekleyeceğimiz katmanları (layers) içe aktaralım.
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense 

# Kullanacağımız optimizer'ı (modeli optimize eden algoritma) tanımlayalım.
from tensorflow.keras.optimizers import Adam 

# Veri setini eğitim ve test kümelerine ayırmak için kullanılır.
from sklearn.model_selection import train_test_split 

import warnings
warnings.filterwarnings("ignore") # Uyarı mesajlarını göz ardı etmek için

# --- Veri Setini Oluştur ---
# user_ids: Kullanıcıların benzersiz kimliklerini (ID'lerini) içeren NumPy dizisi.
# Burada 5 farklı kullanıcı var (0'dan 4'e kadar). 0 numaralı kullanıcı 2 kez alışveriş yapmış gibi ayarlandı.
user_ids = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]) 

# item_ids: Ürünlerin benzersiz kimliklerini (ID'lerini) içeren NumPy dizisi.
# Örneğin, 0 numaralı kullanıcı 0 numaralı ürünü, 4 numaralı kullanıcı 5 numaralı ürünü almış.
item_ids = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 5]) 

# ratings: Kullanıcıların ilgili ürünlere verdiği puanları içeren NumPy dizisi.
# user_ids ve item_ids ile aynı sırayla eşleşir.
ratings = np.array([5, 4, 3, 2, 1, 4, 5, 3, 2, 1]) 


# --- Eğitim ve Test Veri Setlerine Ayırma ---
# Veri setini eğitim (%80) ve test (%20) kümelerine ayırırız.
# random_state: Veri bölme işleminin her seferinde aynı sonuçları vermesi için kullanılır.
user_ids_train, user_ids_test, item_ids_train, item_ids_test, ratings_train, ratings_test = train_test_split(
    user_ids, item_ids, ratings,
    test_size=0.2, # Test veri setinin oranı (toplam verinin %20'si)
    random_state=42 # Yeniden üretilebilirliği sağlar
)


# --- Sinir Ağını Oluşturma Fonksiyonu ---
def create_model(num_users, num_items, embedding_dim):
    """
    Kullanıcı ve ürün etkileşimlerini öğrenmek için basit bir Derin Öğrenme tabanlı tavsiye modeli oluşturur.

    Parametreler:
        num_users (int): Toplam benzersiz kullanıcı sayısı. Embedding katmanının giriş boyutu için kullanılır.
        num_items (int): Toplam benzersiz ürün sayısı. Embedding katmanının giriş boyutu için kullanılır.
        embedding_dim (int): Her bir kullanıcı ve ürün için oluşturulacak embedding vektörlerinin boyutu.
                             Daha yüksek boyut, daha zengin gösterimler sağlayabilir ancak model karmaşıklığını artırır.

    Dönüş:
        tensorflow.keras.models.Model: Derlenmiş Keras modelini döndürür.
    """
    
    # Giriş Katmanları: Kullanıcı ve ürün kimliklerini modelin alacağı giriş noktaları.
    # shape=(1,): Her bir girişte tek bir tam sayı (ID) beklendiğini belirtir.
    user_input = Input(shape=(1,), name="user_input")
    item_input = Input(shape=(1,), name="item_input")

    # Embedding Katmanları: Kullanıcı ve ürün kimliklerini yoğun, düşük boyutlu vektörlere dönüştürür.
    # Bu vektörler, kullanıcıların ve ürünlerin gizli özelliklerini temsil eder.
    # input_dim: Embedding yapılacak benzersiz öğelerin sayısı (örneğin, kullanıcı ID'leri).
    # output_dim: Oluşturulacak embedding vektörünün boyutu (embedding_dim).
    # (user_input) / (item_input): Bu embedding katmanlarının hangi giriş katmanına bağlı olduğunu gösterir.
    user_embedding = Embedding(
        input_dim = num_users, # Kullanıcı ID'lerinin alabileceği maksimum değer + 1 (yani toplam kullanıcı sayısı)
        output_dim = embedding_dim, # Kullanıcı embedding vektörünün boyutu
        name = "user_embedding" # Katmanın adı
        )(user_input) # user_input katmanına bağlanır
    
    item_embedding = Embedding(
        input_dim = num_items, # Ürün ID'lerinin alabileceği maksimum değer + 1 (yani toplam ürün sayısı)
        output_dim = embedding_dim, # Ürün embedding vektörünün boyutu
        name = "item_embedding" # Katmanın adı
        )(item_input) # item_input katmanına bağlanır

    # Vektörleri Düzleştirme (Flatten): Embedding katmanlarından çıkan 2B (batch_size, 1, embedding_dim)
    # tensörleri 1B (batch_size, embedding_dim) vektörlere dönüştürür.
    # Bu, sonraki çarpım işlemi için gereklidir.
    user_vecs = Flatten()(user_embedding)
    item_vecs = Flatten()(item_embedding)
    
    # Dot Product (Nokta Çarpımı): Kullanıcı ve ürün embedding vektörleri arasında nokta çarpımı yapar.
    # Nokta çarpımı, iki vektörün benzerliğini ölçer. Yüksek nokta çarpımı, kullanıcının o ürünü sevme olasılığının
    # yüksek olduğunu gösterir (veya tam tersi).
    # axes=1: Çarpımın son boyutta (vektör boyutu) yapılacağını belirtir.
    dot_product = Dot(axes=1)([user_vecs, item_vecs])
    
    # Çıkış Katmanı: Nokta çarpımının sonucunu tek bir çıkış değerine (tahmin edilen puan) dönüştürür.
    # Dense(1): Bir nöronlu yoğun (fully connected) bir katman.
    output = Dense(1, activation='linear')(dot_product) # Lineer aktivasyon, doğrudan sayısal bir değer (puan) çıktısı verir.
    
    # Modeli Tanımlama: Giriş ve çıkış katmanlarını belirterek Keras modelini oluşturur.
    model = Model(
        inputs= [user_input, item_input], # Modelin beklediği girişler (kullanıcı ve ürün kimlikleri)
        outputs= output # Modelin üreteceği çıkış (tahmin edilen puan)
        )
    
    # Modeli Derleme (Compile): Modelin nasıl eğitileceğini yapılandırır.
    # optimizer: Modelin ağırlıklarını nasıl güncelleyeceğini belirler (örneğin, Adam, SGD).
    #            learning_rate: Optimizer'ın her adımda ağırlıkları ne kadar değiştireceğini belirler.
    # loss: Modelin tahminleri ile gerçek değerler arasındaki farkı ölçmek için kullanılan fonksiyon (hata fonksiyonu).
    #       "mean_squared_error": Ortalama Kare Hata, regresyon problemlerinde yaygın olarak kullanılır.
    model.compile(
        optimizer= Adam(learning_rate = 0.001), # Adam optimizer ve öğrenme oranı
        loss = "mean_squared_error" # Ortalama Kare Hata kaybı
        )
    
    return model

# --- Modeli Eğitme ve Değerlendirme ---

# Toplam benzersiz kullanıcı ve ürün sayısını belirler.
# .max() + 1: ID'ler 0'dan başladığı için maksimum ID'ye 1 eklenerek toplam sayı bulunur.
num_users = user_ids.max() + 1 
num_items = item_ids.max() + 1 

# Embedding vektörlerinin boyutunu belirler.
# Bu, kullanıcı ve ürünlerin özelliklerinin kaç boyutta temsil edileceğini gösterir.
embedding_dim = 8 

# Modeli oluştur.
model = create_model(num_users, num_items, embedding_dim)

# Modeli eğitim veri setleri üzerinde eğit.
# epochs: Eğitim verisi üzerinde kaç tam geçiş yapılacağını belirler. Her epoch'ta model ağırlıkları güncellenir.
# verbose: Eğitim ilerlemesini ne kadar detaylı göstereceğini belirler (1 = ilerleme çubuğu göster).
# validation_split: Eğitim verisinin ne kadarının doğrulama (validation) için ayrılacağını belirler.
#                   Bu kısım eğitim sırasında modelin performansını izlemek için kullanılır, ancak ağırlıklar bu veriyle güncellenmez.
model.fit(
    [user_ids_train, item_ids_train], # Modelin girişleri (kullanıcı ID'leri ve ürün ID'leri)
    ratings_train, # Modelin hedef çıkışları (gerçek puanlar)
    epochs = 20, # 20 eğitim dönemi
    verbose = 1, # Eğitim sürecini detaylı göster
    validation_split = 0.1 # Eğitim verisinin %10'unu doğrulama için ayır
)


# --- Modeli Test Et ---
# Modelin test veri seti üzerindeki performansını değerlendirir.
# Döndürülen 'loss' değeri, modelin test verisindeki ortalama karesel hatasını temsil eder.
loss = model.evaluate([user_ids_test, item_ids_test], ratings_test, verbose=0) # verbose=0, değerlendirme çıktısını gizler
print(f"\nTest Kaybı (Mean Squared Error): {loss:.4f}") # Loss'u 4 ondalık basamakla yazdır

# --- Öneri Testi ---
# Belirli bir kullanıcı ve ürün için tahmin yapma örneği.
user_id_to_predict = np.array([0]) # Tahmin yapılacak kullanıcı ID'si
item_id_to_predict = np.array([5]) # Tahmin yapılacak ürün ID'si

# Modelin tahmin metodunu kullanarak belirli bir kullanıcı-ürün çifti için puanı tahmin et.
prediction = model.predict([user_id_to_predict, item_id_to_predict])

# Tahmin edilen puanı yazdır.
print(f"Kullanıcı ID: {user_id_to_predict[0]}, Ürün ID: {item_id_to_predict[0]} için Tahmin Edilen Puan: {prediction[0][0]:.2f}")