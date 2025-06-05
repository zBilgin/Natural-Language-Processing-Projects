"""
metin üretimi
gpt-2 metin üretimi calismasi
Llama
llma modeli 13.5 gb bu diğer py dosyasında küçük örneği var 
"""

# import libraries
# gpt icin import
from transformers import GPT2LMHeadModel, GPT2Tokenizer 

# lama icin import 
from transformers import AutoTokenizer, AutoModelForCausalLM


# modelin tanimlanmasi
model_name = "gpt2"
#model_name2= "huggyLLama/LLama-7b"

# tokenizer tanimlama ve model olusturma
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#tokenizer_llama = AutoTokenizer.from_pretrained(model_name2)


# Modeli indirme cekme
model = GPT2LMHeadModel.from_pretrained(model_name)
#model_llama = AutoModelForCausalLM.from_pretrained(model_name2)

# metin üretimi icin gerekli olan 
# baslangic text i
text = "Afternoon, "
 

# tokenization
inputs = tokenizer.encode(text, return_tensors="pt")
# return_tensors="pt" paremetresi çıktının
# pytorch tensoru olmasını sağlar

# llama icin tokenization
#inputs_llama = tokenizer_llama(text, return_tensors="pt")

# decoding metin üretimi 

# burada ki inputs modelin baslangic noktasi
# model buradan yola çıkarak başlayacak çalışmasına
#max_length : modelin üreteceği max token sayisi uzunlugu
# bu bize bir output return edecek
outputs = model.generate(
    inputs,
    max_length = 55)

#outputs_llama = model_llama.generate(
 #   inputs.input_ids,
  #  max_length = 55)

# modelin ürettiği token dizisini- tokenları
# tekrar okunabilir bir metne donusturmemiz lazım
# bunun icin tokenizer kullanıp decode yapacağız
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# skip_special_tokens : Cümle basladi mi bittimi burayı daha iyi açıkla
# Ozel tokenleri(örn: cümle başlangıç ve bitiş end of bloc-sentences)
# bize göstereceği metinden cikar 

generated_text_llama = tokenizer.decode(outputs[0], skip_special_tokens=True)

# üretilen metni yazdir
print("Üretilen Metin GPT-2:")
print(generated_text)

print("Üretilen Metin LLma:")
#print(generated_text_llama)



