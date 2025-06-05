from transformers import MarianMTModel, MarianTokenizer
 
# ingilizceden fransızcaya
#model_name = "Helsinki-NLP/opus-mt-en-fr"

# Türkçe'den İngilizce'ye model
model_name = "Helsinki-NLP/opus-mt-tr-en"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Selam, adın ne ?"

# encode edelim, sonrasinda input olarak verelim
translated_text = model.generate(**tokenizer(text, return_tensors="pt", padding=True))


# translated text metne donusturulur
translated_text = tokenizer.decode(translated_text[0], skip_special_tokens=True)

print(f"\nTranslated_text: {translated_text}")


