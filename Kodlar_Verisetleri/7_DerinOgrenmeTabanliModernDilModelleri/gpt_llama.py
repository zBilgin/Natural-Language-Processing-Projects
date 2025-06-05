"""
metin uretimi

gpt-2 metin uretimi calismasi
llama
"""

# import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM # llama

# modelin tanimlanmasi
model_name = "gpt2"
model_name_llama = "huggyllama/llama-7b" # llama


# tokenizer tanimlama ve model olusturma
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer_llama = AutoTokenizer.from_pretrained(model_name_llama) # llama


model = GPT2LMHeadModel.from_pretrained(model_name)
model_llama = AutoModelForCausalLM.from_pretrained(model_name)

# metin uretimi icin gerekli olan baslangic text i
text = "Afternoon, " 

# tokenization
inputs = tokenizer.encode(text, return_tensors="pt")
inputs_llama = tokenizer_llama(text, return_tensors="pt") # llama

# metin uretimi gerceklestirelim
outputs = model.generate(inputs, max_length = 55) # inputs = modelin baslangic noktasi, max_length maximum token sayisi
outputs_llama = model_llama.generate(inputs_llama.input_ids, max_length = 55) # llama

# modelin urettigi tokenlari okunabilir hale getirmemiz lazim
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # ozel tokenleri (orn: cumle baslangic bitis tokenleri) metinden cikart
generated_text_llama = tokenizer.decode(outputs[0], skip_special_tokens=True) # llama
# uretilen metni print ettirelim
print(generated_text)
print(generated_text_llama)

"""
Afternoon,  I was sitting in the kitchen, and I was thinking about the next day's work. 
"""



















