# import library
from transformers import BertTokenizer, BertModel 

import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity


# tokenizer and model create
model_name = "bert-base-uncased" # kucuk boyutlu bert modeli
tokenizer = BertTokenizer.from_pretrained(model_name) # tokenizer yukle
model = BertModel.from_pretrained(model_name) # onceden egitilmis bert modeli

# veri olustur: karsilastirilacak belgeleri ve sorgu cumlemizi olustur

documents = [
    "Machine learning is a field of artificial intelligence",
    "Natural language processing involves understanding human language",
    "Artificial intelligence encomppases machine learning and natural language processing (nlp)",
    "Deep learning is a subset of machine learning",
    "Data science combines statistics, adta analysis and machine learning",
    "I go to shop"
 ]

query = "What is deep learning?" # shopping

# bert ile bilgi getirimi

def get_embedding(text):
    
    # tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # modeli calistir
    outputs = model(**inputs)
    
    # son gizli katmani alalim
    last_hidden_state = outputs.last_hidden_state

    # metni temsili olustur
    embedding = last_hidden_state.mean(dim=1)
    
    # vektoru numpy olarak return et
    return embedding.detach().numpy()

# belgeler ve sorgu icin embedding vektorlerini al
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
query_embedding = get_embedding(query)

# kosinus benzerligi ile belgeler arasinda benzerligi hesaplayalim
similarities = cosine_similarity(query_embedding, doc_embeddings)

# her belgenin benzerlik skoru
for i, score in enumerate(similarities[0]):
    print(f"Document: {documents[i]} \n{score}")

"""
Document: Machine learning is a field of artificial intelligence: 
0.7525447607040405

Document: Natural language processing involves understanding human language: 
0.6778316497802734

Document: Artificial intelligence encomppases machine learning and natural language processing (nlp): 
0.6409367322921753

Document: Deep learning is a subset of machine learning: 
0.7297888994216919

Document: Data science combines statistics, adta analysis and machine learning: 
0.6975172758102417

Document: I go to shop: 
0.5108955502510071
"""

"""
Document: Machine learning is a field of artificial intelligence 
0.634821891784668

Document: Natural language processing involves understanding human language 
0.626939058303833

Document: Artificial intelligence encomppases machine learning and natural language processing (nlp) 
0.5046247243881226

Document: Deep learning is a subset of machine learning 
0.6263622641563416

Document: Data science combines statistics, adta analysis and machine learning 
0.6136887669563293

Document: I go to shop 
0.5354945659637451
"""

most_similar_index = similarities.argmax()

print(f"Most similar document: {documents[most_similar_index]}")






















