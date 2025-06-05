# https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv

"""
problem tanimi
elimizde spam veri seti var ->
spam ve ham ->
binary classification with Decision Tree
"""

# import libraries
import pandas as pd


# veri setini yükle
data = pd.read_csv("../Datasets/Text_Classification_Spam_Dataset.csv", encoding="Latin-1")

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# axis : stop olarak drop inplace: yeni veri setini data-df e eşitle

data.columns = ["Label", "text"]




# EDA: Expretial Data Anaylsis
# Keşifsel veri analizi
print(data.isna().sum())


# %% text cleaning and preprocessing
# özel karekterler, lowercase, 
# tokenization, stopwords, lemmatize

# Gerekli NLTK verilerini indir
# Bu indirmeler, stopwords ve lemmatizasyon süreçleri için gereklidir.
import nltk 

nltk.download("stopwords")

nltk.download("wordnet") 
# 'wordnet' lemmatizer tarafından kelimelerin temel biçimini bulmak için kullanılan bir sözcük veritabanıdır.

nltk.download("omw-1.4") 
# wordnete ait farkli dillerin kelime anlamlarini iceren bir veri seti
# 'omw-1.4' ise çeşitli dilleri destekleyen çok dilli bir wordnet korpusudur.

import re 
from nltk.corpus import stopwords #stop wordlerden kurtulmak icin
from nltk.stem import WordNetLemmatizer # lemmatization

# 'text' sütununu daha kolay yineleme için bir listeye dönüştür
text = list(data.text)

# WordNetLemmatizer'ı başlat
# Lemmatizer'ın bir örneği oluşturulur. Bu nesne, kelimeleri temel veya sözlük biçimlerine dönüştürmek için kullanılacaktır
# (örneğin, "running" kelimesi "run" haline gelir).
lemmatizer = WordNetLemmatizer()

# Temizlenmiş metinleri depolamak için boş bir liste oluştur
corpus = []

for i in range(len(text)):
    
    # Alfabetik olmayan karakterleri kaldır
    # 're.sub("[^a-zA-Z]", " ", text[i])' küçük veya büyük harf İngilizce harf OLMAYAN tüm karakterleri
    # bir boşlukla değiştirir. Bu, sayılar, noktalama işaretleri ve özel sembolleri kaldırır.
    r = re.sub("[^a-zA-Z]"," ", text[i])
    
    r = r.lower()
  
    r = r.split()
    
    # Stopwords'leri kaldır
    # Bu liste anlama, "the", "a", "is" gibi sınıflandırma için genellikle fazla anlam taşımayan
    # yaygın kelimeleri filtreler.
    r = [word for word in r if word not in stopwords.words("english")]
    
    
    # Lemmatizasyon
    # Her kelime temel biçimine dönüştürülür. Bu, kelime dağarcığı boyutunu azaltmaya yardımcı olur
    # ve bir kelimenin farklı çekimlerinin aynı şekilde ele alınmasını sağlar
    # (örneğin, "caring", "cares", "cared" hepsi "care" olur).
    r = [lemmatizer.lemmatize(word) for word in r] 
    
    
    # İşlenmiş kelimeleri tekrar tek bir dizeye birleştir
    # Temizlenmiş ve lemmatize edilmiş kelimeler, tutarlı bir dize oluşturmak için boşluklarla birleştirilir.
    r = " ".join(r)
    
    # Temizlenmiş metni corpus listesine ekle
    corpus.append(r)
    
data["text2"] = corpus
    
# %% model training and evulation

X = data["text2"]
y = data["Label"] # target variable


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=42)

# Feature extraction: bag of words
# CountVectorizer Metin verisini sayısal özellik vektörlerine dönüştürür.
# Özellik çıkarımı: Bag of Words (BOW)
# CountVectorizer, metin verisini makine öğrenimi modellerinin anlayabileceği sayısal bir biçime dönüştürmek için kullanılır.
# Her satırın bir belge (metin mesajı) ve her sütunun sözlükteki benzersiz bir kelimeyi temsil ettiği
# bir matris oluşturur, hücre değeri o kelimenin belgedeki sıklığını gösterir.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

# 'fit_transform' eğitim verisinden kelime dağarcığını öğrenir ve ardından
# eğitim metnini sayısal bir matrise dönüştürür.
X_train_cv = cv.fit_transform(X_train)

# classifier training: model training and evulation
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# modeli eğitme
dt.fit(X_train_cv, y_train) # eğitim


# Test verisini eğitim verisine göre eğitilmiş *aynı* CountVectorizer'ı kullanarak dönüştür
# Aynı kelime dağarcığının ve özellik eşlemesinin uygulandığından emin olmak için
# test verisini yalnızca 'transform' etmek, 'fit_transform' etmemek çok önemlidir.
X_test_cv = cv.transform(X_test)


# prediction # Test seti üzerinde tahminler yap
prediction = dt.predict(X_test_cv)

# sonucları karşılaştırma tahmin-gerçek
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(y_test, prediction)

# Karışıklık matrisinden doğruluk (accuracy) hesapla
# Doğruluk, doğru pozitiflerin (TP) ve doğru negatiflerin (TN) toplamının
# toplam örnek sayısına bölünmesiyle hesaplanır.
# c_matrix[0,0] = Gerçek Negatifler (TN) - Doğru sınıflandırılmış ham mesajlar
# c_matrix[1,1] = Gerçek Pozitifler (TP) - Doğru sınıflandırılmış spam mesajları
# c_matrix[1,0] = Yanlış Negatifler (FN) - Yanlışlıkla ham olarak sınıflandırılmış spam mesajları
# c_matrix[0,1] = Yanlış Pozitifler (FP) - Yanlışlıkla spam olarak sınıflandırılmış ham mesajları
# Sağlanan hesaplamada:
# (sum(sum(c_matrix))) toplam örnek sayısıdır.
# (c_matrix[1,0] + c_matrix[0,1]) toplam yanlış sınıflandırmaları (FN + FP) temsil eder.
# (sum(sum(c_matrix)) - (c_matrix[1,0] + c_matrix[0,1])) etkin bir şekilde (TN + TP) değerini hesaplar.
# confusion matrix üzerinde tp diğer değerler ile başarı hesaplama
accuracy = 100*(sum(sum(c_matrix)) - c_matrix[1,0] - c_matrix[0,1] ) / sum(sum(c_matrix) )

print(f"Başarım: {accuracy}")












