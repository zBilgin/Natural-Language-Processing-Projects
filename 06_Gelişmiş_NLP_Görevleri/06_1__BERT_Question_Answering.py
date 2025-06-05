from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

# Kullanacağımız bert modeli 
# Squat veri seti üzerinde eğitilmiş 
# Fine tuning yapılmış büyük bir bert dil modeli

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# bert tokenizer 
tokenizer = BertTokenizer.from_pretrained(model_name)

# soru cevaplama görevi için ince ayar yapılmış bert modeli
model = BertForQuestionAnswering.from_pretrained(model_name)


# cevaplari tahmin eden fonksiyon
def predict_answer(context, question):
    """
    Parameters
    ----------
    context : metin.
    question : soru.
    Amac-Goal: Metnin icerisinden soruyu bulmak
    
    1- Tokenize- elimiz de bulunan quesiton ve metni tokenize et
    2- Metnin icerisinde soruyu ara bu bize neyi return etti
    3- Metnin icerisinde sorunun cevabının nereler de olabileceğinin skolarini  return etti
    4- Bu skorlardan tokenaların indexlerini hesapladık
    5- Tokenları bulduk yani cevabı bulduk
    6- Okunabilir olması icin tokenalardan stringe cevirdik
    Returns
    -------
    answer : TYPE
        DESCRIPTION.

    """
    
    
    
    # metni ve soruyu tokenlere ayirma 
    # ve modele uygun hale getirme
    # aslında bildiğimiz  tokenleştirme
    # işlemi yapıyoruz ama bu sefer berd ile
    encoding = tokenizer.encode_plus(
    question,
    context,
    return_tensors="pt", 
    max_length=512,
    truncation=True
    )
    # return tensorda ki pt tensor anlamına geliyor, çıktımız pytorch tensorları formatında 
    # max_length modelin işleyebileceği maksimum token sayısı
    # truncation=true token sayısı 512 geçerse kesilsin true yaptık
    
    
    # Giris tensorlari hazirla
    
    # tokenlerin idleri
    input_ids = encoding["input_ids"] 

    #hangi tokenların dikkate alınacağını belirtiyor
    attention_mask = encoding["attention_mask"] 

    # modeli calistir ve skorlari hesapla
    
    # with torch.no_grad dediğimiz kavram
    # hesaplama sırasında gradyanların hesaplanmasını devre dışı bırakıyor
    # buda daha hızlı hesaplanmasınıo sağlıyor daha az bellek kullanıyor 
    with torch.no_grad():
        start_scores, end_scores = model(
            input_ids,
            attention_mask = attention_mask,
            return_dict = False            
            )
    # burada start scores ve end scores cevabın metni içerisinde
    # metin içerisinde nerede başladığını bittiğini belirten scorelar
    
    
    # en yuksek olasiliğa sahip start ve end indekslerini hesaplama
    start_index = torch.argmax(start_scores, dim=1).item() # baslangic indeximiz
    end_index = torch.argmax(end_scores, dim=1).item() # bitis indeximiz
    
    # token id lerini kullanarak cevap metnini elde edelim
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index + 1])
    
    # tokenleri birlestir ve okunabilir hale getir
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer 

question = "What is the capital of France"
context = "France, officallt the French Republic, is a country whose capitaL is Paris"

answer = predict_answer(context, question)
print(f"Answer : {answer}")


question = "What is Machine Learning?"
context = ''' Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance 
                on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or 
                decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection 
                of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning 
                is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, 
                theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory 
                data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics. '''


answer = predict_answer(context, question)
print(f"Answer: {answer}")
