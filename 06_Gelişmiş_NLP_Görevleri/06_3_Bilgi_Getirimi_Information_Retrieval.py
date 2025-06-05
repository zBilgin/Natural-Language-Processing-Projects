# --- Import necessary libraries ---
# --- Gerekli kütüphaneleri içe aktar---

# Hugging Face's Transformers library for BERT
# Hugging Face'in BERT için Transformers kütüphanesi
from transformers import BertTokenizer, BertModel

import numpy as np

# For calculating cosine similarity between vectors
# Vektörler arasındaki kosinüs benzerliğini hesaplamak için
from sklearn.metrics.pairwise import cosine_similarity


# --- Tokenizer and Model Creation ---
# --- Tokenizer ve Model Oluşturma ---

# Define the name of the pre-trained BERT model.
# "bert-base-uncased" is a smaller, uncased (case-insensitive) BERT model.
# Uncased means it treats "Machine Learning" and "machine learning" as the same.
# Önceden eğitilmiş BERT modelinin adını tanımlayın.
# "bert-base-uncased" daha küçük, büyük/küçük harf duyarsız bir BERT modelidir.
# "Uncased" (küçük harfli), "Machine Learning" ve "machine learning" ifadelerini aynı kabul ettiği anlamına gelir.

model_name = "bert-base-uncased" 

# Load the tokenizer corresponding to the chosen model.
# The tokenizer converts text into numerical IDs that the BERT model can understand.
# Seçilen modele karşılık gelen tokenizer'ı yükle.
# Tokenizer, metni BERT modelinin anlayabileceği sayısal ID'lere dönüştürür.
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load the pre-trained BERT model itself.
# This model has been trained on a massive amount of text data and understands language nuances.
# Önceden eğitilmiş BERT modelini yükle.
# Bu model büyük miktarda metin verisi üzerinde eğitilmiştir ve dilin nüanslarını anlar.
model = BertModel.from_pretrained(model_name)


# --- Data Preparation ---
# --- Veri Hazırlığı ---

# Define the documents that we want to search through.
# These are the pieces of text against which our query will be compared.
# Üzerinde arama yapmak istediğimiz belgeleri tanımlayın.
# Bunlar, sorgumuzun karşılaştırılacağı metin parçalarıdır.
documents = [
    "Machine Learning is a field of artificial intelligence",
    "Natural Languange processing involves understanding human language",
    "Artfical intelligence  encomppases machine Learning and natural language processing (nlp)",
    "Deep Learning is a subset of machine learning",
    "Data science combines statistics, data anaLysis and machine Learning",
    "I go to shop"
    ]

# Define the query sentence.
# This is the question or statement we want to find similar documents for.
# Sorgu cümlesini tanımlayın.
# Bu, benzer belgeler bulmak istediğimiz soru veya ifadedir.
query = "What is deep learning?"
# You can uncomment and test with other queries:
# Diğer sorgularla test etmek için yorum satırını kaldırabilirsiniz:
# query = "What is machine Learning?"
# query = "shoping"


# --- BERT-based Information Retrieval Function ---
# --- BERT Tabanlı Bilgi Getirme Fonksiyonu ---

def get_embedding(text):
    """
    Generates a numerical vector (embedding) for a given text using the BERT model.
    This embedding captures the semantic meaning of the text.
    BERT modelini kullanarak verilen bir metin için sayısal bir vektör (embedding) oluşturur.
    Bu embedding, metnin anlamsal anlamını yakalar.

    Args:
    text (str): The input text (document or query) to be embedded.
    text (str): Gömülecek girdi metni (belge veya sorgu).
        
    Returns:
    numpy.ndarray: A 1D NumPy array representing the embedding of the input text.
    numpy.ndarray: Girdi metninin embedding'ini temsil eden 1D NumPy dizisi.
"""
    
    # Tokenization: Convert the input text into numerical input IDs, attention masks, etc.,
    # that the BERT model expects.
    # return_tensors="pt": Returns PyTorch tensors.
    # truncation=True: Truncates the input if it's longer than the model's maximum sequence length.
    # padding=True: Pads the input to the maximum sequence length or the longest sequence in the batch.
    # Tokenizasyon: Girdi metnini, BERT modelinin beklediği sayısal girdi ID'lerine, dikkat maskelerine vb. dönüştürür.
    # return_tensors="pt": PyTorch tensörlerini döndürür.
    # truncation=True: Girdi, modelin maksimum dizi uzunluğundan daha uzunsa kırpılır.
    # padding=True: Girdi, maksimum dizi uzunluğuna veya batch'teki en uzun diziye kadar doldurulur.
    
    inputs = tokenizer(
        text,
        return_tensors="pt", # "pt" stands for PyTorch tensors
        truncation=True,
        padding=True
        )

    # Run the BERT model with the tokenized inputs.
    # The double asterisk (**) unpacks the dictionary of inputs into keyword arguments.
    # This is a Python syntax for passing dictionary items as named arguments to a function.
    # Tokenize edilmiş girdilerle BERT modelini çalıştırın.
    # Çift yıldız (**) girdiler sözlüğünü anahtar kelime argümanlarına açar.
    # Bu, bir sözlük öğelerini bir fonksiyona adlandırılmış argümanlar olarak iletmek için bir Python sözdizimidir.
    outputs = model(**inputs)
    
    # Get the last hidden state from the model's output.
    # The 'last_hidden_state' is a tensor containing the contextualized embeddings for each token
    # in the input sequence. It represents the final output of the BERT model's layers.
    # We are interested in the last hidden state because it captures the most refined
    # semantic information after processing through all layers of the BERT model.
    # Modelin çıktısından son gizli katmanı (last hidden state) alın.
    # 'last_hidden_state', girdi dizisindeki her token için bağlamsallaştırılmış embedding'leri içeren bir tensördür.
    # BERT modelinin tüm katmanlarından geçtikten sonraki nihai çıktıyı temsil eder.
    # En son gizli katmanla ilgileniyoruz çünkü BERT modelinin tüm katmanlarından geçtikten sonra
    # en rafine anlamsal bilgiyi yakalar.
    last_hidden_state = outputs.last_hidden_state
    
    
    # To get a single representative vector for the entire text (sentence embedding),
    # we average the embeddings of all tokens in the 'last_hidden_state'.
    # dim=1: Specifies that the mean should be calculated across the token dimension,
    # effectively averaging the embeddings of all tokens for each sequence in the batch.
    # Tüm metin için tek bir temsilci vektör (cümle embedding'i) elde etmek için,
    # 'last_hidden_state' içindeki tüm tokenların embedding'lerinin ortalamasını alırız.
    # dim=1: Ortalama almanın token boyutu boyunca yapılması gerektiğini belirtir,
    # bu da batch'teki her dizi için tüm tokenların embedding'lerinin ortalamasını alır.
    embedding = last_hidden_state.mean(dim=1)
    
    # Detach the tensor from the computation graph and convert it to a NumPy array.
    # .detach(): Creates a new tensor that does not require gradients. This is important
    # because we are not training the model here; we just want the numerical value.
    # .numpy(): Converts the PyTorch tensor to a NumPy array for easier manipulation.
    # Tensörü hesaplama grafiğinden ayırın ve bir NumPy dizisine dönüştürün.
    # .detach(): Gradyan gerektirmeyen yeni bir tensör oluşturur. Bu, modeli burada eğitmediğimiz için önemlidir;
    # sadece sayısal değeri istiyoruz.
    # .numpy(): PyTorch tensörünü daha kolay işleme için bir NumPy dizisine dönüştürür.
    return embedding.detach().numpy()



# --- Generate Embeddings for Documents and Query ---
# --- Belgeler ve Sorgu İçin Embedding'ler Oluştur ---

# Get the embedding vector for each document and stack them vertically.
# np.vstack: Stacks arrays in sequence vertically (row wise).
# This creates a 2D array where each row is the embedding of a document.
# Her belge için embedding vektörünü alın ve bunları dikey olarak istifleyin.
# np.vstack: Dizileri dikey olarak (satır bazında) sırayla istifler.
# Bu, her satırın bir belgenin embedding'i olduğu 2B bir dizi oluşturur.
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])


# Get the embedding vector for our query.
# Sorgumuz için embedding vektörünü alın.
query_embedding = get_embedding(query)


# --- Calculate Similarities and Find Most Similar Document ---
# --- Benzerlikleri Hesapla ve En Benzer Belgeyi Bul ---

# Calculate the cosine similarity between the query embedding and each document embedding.
# Cosine similarity measures the cosine of the angle between two vectors.
# A higher cosine similarity (closer to 1) indicates greater similarity in direction
# (and thus, semantic meaning) between the vectors.
# Sorgu embedding'i ile her belge embedding'i arasındaki kosinüs benzerliğini hesaplayın.
# Kosinüs benzerliği, iki vektör arasındaki açının kosinüsünü ölçer.
# Daha yüksek bir kosinüs benzerliği (1'e daha yakın), vektörler arasında daha fazla yön benzerliği (ve dolayısıyla anlamsal anlam) olduğunu gösterir.
similarities = cosine_similarity(query_embedding, doc_embeddings)

# Print the similarity score for each document.
# similarities[0] is used because cosine_similarity returns a 2D array,
# even if there's only one query. We access the first (and only) row.
# Her belgenin benzerlik skorunu yazdır.
# similarities[0] kullanılır çünkü cosine_similarity, yalnızca bir sorgu olsa bile 2B bir dizi döndürür.
# İlk (ve tek) satıra erişiriz.
print("\n--- Similarity Scores ---")
for i, score in enumerate(similarities[0]):
    print(f"Document: \"{documents[i]}\"")
    print(f"Similarity Score: {score}\n") 

# Find the index of the document with the highest similarity score.
# En yüksek benzerlik skoruna sahip belgenin dizinini bulun.
most_similar_index = similarities.argmax()

# Print the most similar document based on the calculated cosine similarities.
# Hesaplanan kosinüs benzerliklerine göre en benzer belgeyi yazdırın.
print("--- Most Similar Document ---")
print(f"Most similar document: {documents[most_similar_index]}")
print(f"Similarity Score: {similarities[0][most_similar_index]}")