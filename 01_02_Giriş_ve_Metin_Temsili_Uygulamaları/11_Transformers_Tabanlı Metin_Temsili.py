# -*- coding: utf-8 -*-
"""
Created on Sat May 17 19:59:59 2025

@author: Zekeriya
"""

# ==============================================
# 🧠 Transformers Tabanlı Metin Temsili (BERT)
# Model: bert-base-uncased (Hugging Face Transformers)
# ==============================================

# 📦 Gerekli Kütüphaneler
from transformers import AutoTokenizer, AutoModel
import torch

# ==============================================
# 📥 Model ve Tokenizer Yükleme

# "bert-base-uncased" adlı önceden eğitilmiş modeli kullanıyoruz.
# Bu model İngilizce cümleleri küçük harfe çevirerek işler (uncased).
model_name = "bert-base-uncased"

# Tokenizer: Metni BERT’in anlayacağı token formatına çevirir
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Model: BERT modelini yükler (transformer encoder blokları içerir)
model = AutoModel.from_pretrained(model_name)

# ==============================================
# 📝 Metin Girişi
text = "Transformers can be used for natural language processing."

# Tokenizasyon işlemi
# return_tensors="pt" → sonucu PyTorch tensörü (tensor) olarak döndürür
inputs = tokenizer(text, return_tensors="pt")

# ==============================================
# ⚙️ Model ile Gösterim (Embedding) Elde Etme

# torch.no_grad(): Eğitim değil, sadece çıkarım (inference) yapacağımız için
# bellekte yer tutacak gradyan hesaplamasını kapatır.
# gradyanların hesaplaması durdurulur
# böylece belleği daha verimli kullanırız
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: Tüm tokenlar için son katmandaki vektör çıktısı
# Şekil: [batch_size, sequence_length, hidden_size]
last_hidden_state = outputs.last_hidden_state

# İlk token'in (CLS token) embedding’ini al
first_token_embedding = last_hidden_state[0, 0, :].numpy()

# Sonucu yazdır
print("🔹 Metin Temsili (CLS token embedding):")
print(first_token_embedding)
