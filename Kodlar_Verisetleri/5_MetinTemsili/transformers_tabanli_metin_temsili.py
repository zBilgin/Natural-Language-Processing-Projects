# import libraries
from transformers import AutoTokenizer, AutoModel
import torch

# model ve tokenizer yukle
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# input text (metni) tanimla
text = "Transformers can be used for natural language processing."

# metni tokenlara cevirmek
inputs = tokenizer(text, return_tensors = "pt") # cikti pytorch tensoru olarak return edilir

# modeli kullanarak metin temsili olustur
with torch.no_grad(): # gradyanlarin hesaplanmasi durdurulur, boylece bellegi daha verimli kullaniriz.
    outputs = model(**inputs)

# modelin cikisindan son gizli durumu alalim
last_hidden_state = outputs.last_hidden_state # tum token ciktilarini almak icin

# ilk tokenin embedding inin alalim ve print ettirelim
first_token_embedding = last_hidden_state[0,0,:].numpy()

print(f"Metin temsili: {first_token_embedding}")



