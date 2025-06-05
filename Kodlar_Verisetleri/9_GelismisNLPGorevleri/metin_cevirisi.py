from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr" # eng to fr --- "Helsinki-NLP/opus-mt-fr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Hello, what is your name"

# encode edelim, sonrasinda modele input olarak verelim
translated_text = model.generate(**tokenizer(text, return_tensors="pt", padding = True))

# translated text metne donusturulur
translated_text = tokenizer.decode(translated_text[0], skip_special_tokens=True)
print(f"Translated_text: {translated_text}")
