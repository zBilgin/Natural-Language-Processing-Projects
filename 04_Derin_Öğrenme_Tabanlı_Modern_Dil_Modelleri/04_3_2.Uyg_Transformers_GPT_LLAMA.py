# Gerekli kütüphanelerin import edilmesi
# Metin üretimi için Hugging Face Transformers kütüphanesini kullanacağız.
# Bu kütüphane, önceden eğitilmiş birçok dil modeline ve tokenizasyon araçlarına kolay erişim sağlar.

# GPT-2 modeli için gerekli modüllerin import edilmesi
# GPT2LMHeadModel: GPT-2 modelinin metin üretimi için kullanılan başlık (head) kısmını içeren sınıf.
#                    Bu sınıf, bir sonraki token'ı tahmin etmek üzere tasarlanmıştır.
# GPT2Tokenizer: GPT-2 modeli için özel olarak eğitilmiş tokenizasyon aracı.
#                Metni modelin anlayabileceği token ID'lerine dönüştürür ve tam tersini yapar.
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Alternatif (Llama benzeri) model için gerekli modüllerin import edilmesi
# AutoTokenizer: Model adına göre otomatik olarak uygun tokenizasyon aracını yükler.
#                Farklı modeller farklı tokenizasyon yöntemleri kullanabilir, bu sınıf bu süreci basitleştirir.
# AutoModelForCausalLM: Nedensel dil modellemesi (Causal Language Modeling) için otomatik olarak uygun modeli yükler.
#                       Bu tür modeller, bir sonraki kelimeyi veya token'ı tahmin etmek için kullanılır (GPT ve Llama gibi).
from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch # Eğer 4-bit/8-bit yükleme için BitsAndBytesConfig kullanacaksanız torch importu gerekebilir
# from transformers import BitsAndBytesConfig # İsteğe bağlı: Daha fazla bellek tasarrufu için

# Kullanılacak modellerin tanımlanması
# model_name: GPT-2 modelinin Hugging Face Model Hub'daki kimliği.
#             "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl" gibi farklı boyutları mevcuttur.
#             Burada standart "gpt2" modeli kullanılıyor.
model_name_gpt2 = "gpt2"

# model_name_alternative: Daha hafif bir Llama benzeri modelin Hugging Face Model Hub'daki kimliği.
#                         "TinyLlama/TinyLlama-1.1B-Chat-v1.0" modeli, yaklaşık 1.1 milyar parametreye sahip olup,
#                         bellek kullanımı açısından daha verimlidir (FP16'da ~2.2GB, nicelleştirme ile daha da az).
#                         Bu, 1-2 GB bellek hedefi için iyi bir başlangıç noktasıdır.
#                         Alternatif olarak "microsoft/phi-3-mini-4k-instruct" (3.8B parametre, 4-bit nicelleştirme ile ~1.9GB)
#                         da düşünülebilir ve çok yeteneklidir.
model_name_alternative = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name_alternative = "microsoft/phi-3-mini-4k-instruct" # Başka bir güçlü ve küçük alternatif

# Tokenizer'ların tanımlanması ve modellerin oluşturulması

# GPT-2 için tokenizer'ın yüklenmesi
# GPT2Tokenizer.from_pretrained(model_name) ile belirtilen model için önceden eğitilmiş
# tokenizer (sözlük ve kurallar) indirilir ve yüklenir.
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(model_name_gpt2)

# Alternatif model için tokenizer'ın yüklenmesi
# AutoTokenizer.from_pretrained(model_name_alternative) ile belirtilen model için
# uygun tokenizer otomatik olarak indirilir ve yüklenir.
tokenizer_alternative = AutoTokenizer.from_pretrained(model_name_alternative)

# Modellerin indirilmesi ve yüklenmesi

# GPT-2 modelinin yüklenmesi
# GPT2LMHeadModel.from_pretrained(model_name) ile belirtilen modelin ağırlıkları
# (weights) Hugging Face Hub'dan indirilir ve model mimarisi oluşturulur.
model_gpt2 = GPT2LMHeadModel.from_pretrained(model_name_gpt2)

# Alternatif (TinyLlama) modelinin yüklenmesi
# AutoModelForCausalLM.from_pretrained(model_name_alternative) ile belirtilen modelin
# ağırlıkları indirilir ve nedensel dil modeli olarak yüklenir.
# Bu işlem, modelin boyutuna bağlı olarak zaman ve disk alanı gerektirebilir.
# Daha fazla bellek tasarrufu için nicelleştirme (quantization) düşünülebilir.
# Örneğin, `bitsandbytes` kütüphanesi ile 4-bit yükleme:
# bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4", # veya "fp4"
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16 # veya torch.float16
# )
# model_alternative = AutoModelForCausalLM.from_pretrained(
#     model_name_alternative,
#     quantization_config=bnb_config, # 4-bit yükleme için
#     device_map="auto" # Modeli uygun aygıta (GPU/CPU) otomatik dağıtır
# )
# Eğer nicelleştirme kullanmıyorsanız veya basit tutmak istiyorsanız:
model_alternative = AutoModelForCausalLM.from_pretrained(model_name_alternative)


# Metin üretimi için başlangıç metni (prompt)
# Modelin metin üretimine başlaması için verilecek olan ilk girdi.
# Model, bu metni devam ettirmeye çalışacaktır.
text = "Afternoon" # Afternoon, -> Öğleden sonra, (Türkçe örnek)

# GPT-2 için metnin tokenizasyonu
# tokenizer.encode(), verilen metni modelin anlayabileceği token ID'lerine dönüştürür.
# return_tensors="pt": Bu parametre, çıktının PyTorch tensörleri formatında olmasını sağlar.
#                      TensorFlow için "tf", NumPy dizileri için "np" kullanılabilir.
#                      Hugging Face modelleri genellikle PyTorch veya TensorFlow ile çalışır.
inputs_gpt2 = tokenizer_gpt2.encode(text, return_tensors="pt")

# Alternatif model için metnin tokenizasyonu
# Aynı şekilde, alternatif modelin tokenizer'ı kullanılarak başlangıç metni token ID'lerine dönüştürülür.
# inputs_alternative bir dictionary dönecektir, genellikle 'input_ids' ve 'attention_mask' içerir.
inputs_alternative = tokenizer_alternative(text, return_tensors="pt")

# Metin üretimi (Decoding / Generation)

# GPT-2 ile metin üretimi
# model.generate() fonksiyonu, verilen input token'larından başlayarak yeni token'lar üretir.
# inputs_gpt2: Modelin üretime başlayacağı token ID'leri.
# max_length: Üretilecek metnin maksimum token sayısı (başlangıç metni dahil).
#             Bu değer, üretilen metnin uzunluğunu sınırlar.
#             Daha uzun metinler için bu değer artırılabilir, ancak bu işlem süresini de artırır.
#             Diğer popüler parametreler:
#               num_beams: Beam search için kullanılacak ışın sayısı. Daha kaliteli ama yavaş üretim sağlar.
#               no_repeat_ngram_size: Tekrar eden n-gram'ları engeller.
#               top_k, top_p (nucleus sampling): Üretim sırasında olasılığı düşük token'ları filtreler.
#               temperature: Üretimin rastgeleliğini ayarlar. Düşük değerler daha deterministik,
#                            yüksek değerler daha rastgele ve yaratıcı sonuçlar verir.
outputs_gpt2 = model_gpt2.generate(
    inputs_gpt2,
    max_length=55,
    num_return_sequences=1 # Kaç farklı metin sekansı üretileceği
)

# Alternatif model ile metin üretimi
# inputs_alternative.input_ids: Tokenize edilmiş girdinin 'input_ids' kısmını modele veriyoruz.
#                         Eğer tokenizer doğrudan tensor döndürdüyse ve bu tensor input_ids ise
#                         doğrudan o da verilebilir. AutoTokenizer'dan dönen dictionary'nin
#                         içindeki 'input_ids' tensörünü kullanmak daha genel bir yaklaşımdır.
# Not: Bazı modeller (özellikle chat/instruct modelleri) için özel prompt formatları gerekebilir.
# TinyLlama-1.1B-Chat-v1.0 için, eğer sohbet formatında kullanacaksanız, prompt'u ona göre düzenlemeniz gerekebilir.
# Örnek basit kullanım:
# Eğer model GPU'daysa ve input CPU'daysa, input'u GPU'ya taşımanız gerekebilir:
# input_ids_on_device = inputs_alternative.input_ids.to(model_alternative.device)
# attention_mask_on_device = inputs_alternative.attention_mask.to(model_alternative.device)
# outputs_alternative = model_alternative.generate(
#     input_ids_on_device,
#     attention_mask=attention_mask_on_device,
#     max_length=55,
#     num_return_sequences=1
# )
# Basit hali (model ve input aynı cihazdaysa veya device_map="auto" kullanıldıysa):
outputs_alternative = model_alternative.generate(
    inputs_alternative.input_ids,
   
    max_length=55,

)

# Üretilen token dizilerinin okunabilir metne dönüştürülmesi (Decoding)

# GPT-2 için decoding
# tokenizer.decode() fonksiyonu, model tarafından üretilen token ID'lerini tekrar insan tarafından okunabilir metne çevirir.
# outputs_gpt2[0]: model.generate() fonksiyonu genellikle bir liste içinde tensörler döndürür.
#                  Eğer num_return_sequences=1 ise, ilk (ve tek) üretilen sekansı alırız.
# skip_special_tokens=True: Bu parametre, üretim sırasında kullanılan özel token'ların (örneğin,
#                             [CLS], [SEP], [PAD], <|endoftext|> gibi cümle başı/sonu, dolgu token'ları)
#                             son metinden çıkarılmasını sağlar. Bu, metni daha temiz ve okunabilir hale getirir.
generated_text_gpt2 = tokenizer_gpt2.decode(outputs_gpt2[0], skip_special_tokens=True)

# Alternatif model için decoding
# Aynı şekilde, alternatif model tarafından üretilen token'lar da kendi tokenizer'ı ile okunabilir metne dönüştürülür.
generated_text_alternative = tokenizer_alternative.decode(outputs_alternative[0], skip_special_tokens=True)

# Üretilen metinlerin yazdırılması
print("--- Üretilen Metin GPT-2: ---")
print(generated_text_gpt2)

print("\n--- Üretilen Metin (Alternatif Model - TinyLlama): ---")
print(generated_text_alternative)