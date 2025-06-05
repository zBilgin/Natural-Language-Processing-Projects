"""
varlik ismi tanima: metin(cümle) ->
metin icerisinde bulunan varlik isimlerini tanimla
"""

# import libraries 
import pandas as pd
import spacy 

# spaCy'nin İngilizce dil modelini yükle
# 'en_core_web_sm' modeli, İngilizce metinler üzerinde eğitilmiş, küçük boyutlu ve genel amaçlı bir modeldir.
# Bu model, kelime vektörleri, parçalı konuşma etiketleme (POS tagging), bağımlılık ayrıştırma (dependency parsing)
# ve varlık tanıma (NER) gibi birçok NLP görevini gerçekleştirebilir.
# Eğer bu model sisteminizde yüklü değilse, terminalinizde 'python -m spacy download en_core_web_sm' komutunu çalıştırmanız gerekir.
nlp = spacy.load("en_core_web_sm")
 

# Analiz edilecek örnek metin
# Bu metin, içinde çeşitli türde varlıklar (kişi, organizasyon, yer, tarih vb.) barındırır.
content = "Alice works at Amazon and lives in London. She visited the British Museum last weekend."

# spaCy modelini metin üzerinde çalıştır
# 'nlp(content)' fonksiyonu, metni işler ve bir 'Doc' (doküman) nesnesi oluşturur.
# Bu 'Doc' nesnesi, metnin tüm işlenmiş halini (tokenler, etiketler, varlıklar vb.) içerir.
# Bu işlem, metindeki varlıkları (entities) analiz eder ve tanımlar.
doc = nlp(content)
# bu islem metindeki varliklari (entities) analiz eder

# 'doc.ents' özelliği, spaCy tarafından tanınan tüm varlıkların bir listesini içerir.
for ent in doc.ents:
    #ent.text: varlık ismi (Alice, Amazon)
    #ent.start_char ve ent.end_char : varliğin metindeki baslangic ve bitis karakterleri
    #ent.label_ : varlık türü alice -> kişi amazon -> organizasyon
    # ent.text: Tanımlanan varlığın metin değeri (örneğin "Alice", "Amazon").
    # ent.start_char: Varlığın orijinal metindeki başlangıç karakter indeksi.
    # ent.end_char: Varlığın orijinal metindeki bitiş karakter indeksi.
    # ent.label_: spaCy tarafından atanan varlık türü etiketi (örneğin "PERSON", "ORG", "GPE").
    # ORG: Organization (Kuruluş), GPE: Geopolitical Entity (Jeopolitik Varlık - Ülke, Şehir vb.), PERSON: Kişi.
    print(
        ent.text, 
        ent.start_char,
        ent.end_char,
        ent.label_)


# Ek bilgi: 'ent.lemma_' özelliği (sadece tokenler için geçerlidir)
# Varlıklar (ent.text) birden fazla kelimeden oluşabileceği için doğrudan 'ent.lemma_' kullanılamaz.
# Ancak eğer tek bir kelime (token) üzerinden lemmatizasyon yapmak isterseniz, 'token.lemma_' kullanırdınız.
# Örneğin, 'last weekend' bir varlıktır ama bunun tek bir kök kelimesi olmaz.
# Bu nedenle, bu projede varlık için doğrudan 'ent.lemma_' kullanımı uygun değildir.
# Ancak eğer her bir kelimenin (token'ın) kök halini almak isterseniz:
# for token in doc:
#     print(token.text, token.lemma_)

# ent.lema: varliğin kök hali
entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]

# varlik listesini pandas df e çevir
# varlık isimleri çok olduğunda okumak daha iyi olur
df = pd.DataFrame(entities, columns=["text", "type", "lemma"])

