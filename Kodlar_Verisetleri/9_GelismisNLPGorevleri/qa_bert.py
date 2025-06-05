from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

# squad veri seti uzerinde ince ayar yapilmis bert fil modeli
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# bert tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# soru cevaplama gorevi icin ince ayar yapilmis bert modeli
model = BertForQuestionAnswering.from_pretrained(model_name)

# cevaplari tahmin eden fonksiyon
def predict_answer(context, question):
    """
        context = metin
        question = soru
        Amac: metin icerisinden soruyu bulmak
        
        1) tokenize
        2) metnin icerisinde soruyu ara
        3) metnin icerisinde sorunun cevabinin nerelerde olabileceginin skorlarini return etti
        4) skorlardan tokenlarin indeksleri hesapladik
        5) tokenlari bulduk yani cevabi bulduk
        6) okunabilir olmasi icin tokenlardan string'e cevirdik
    """
    
    # metni ve soruyu tokenlara ayiralim ve modele uygun hale getirelim
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # giris tensorlerini hazirla
    input_ids = encoding["input_ids"] # tokenlerin id
    attention_mask = encoding["attention_mask"] # hangi tokenlarin dikkate alinacagini belirtir
    
    # modeli calistir ve skorlari hesapla
    with torch.no_grad():
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict = False)
        
    # en yuksek olasiliga sahip start ve end indekslerini hesapliyor
    start_index = torch.argmax(start_scores, dim=1).item() # baslangic indeks
    end_index = torch.argmax(end_scores, dim=1).item() # bitis indeksimiz
    
    # token id lerini kullanarak cevap metinin elde edelim
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index + 1])
    
    # tokenlari birlestir ve okunabilir hale getir
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer
    
question = "What is the capital of France"
context = "France, officially the French Republic, is a country whose capital is Paris"

answer = predict_answer(context, question)
print(f"Answer: {answer}")

question = '''What is Machine Learning?'''
context = ''' Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance 
                on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or 
                decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection 
                of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning 
                is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, 
                theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory 
                data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics. '''


answer = predict_answer(context, question)
print(f"Answer: {answer}")




















