import nltk

from nltk.corpus import stopwords
# Bu nltk verisetinden gerekli corpus modulunden stopwords classı gibi


# bu veriseti stopword veri seti 
# yaygın olanları farklı dillerde ki en cok kullaılanlar
nltk.download("stopwords")


# ingilizce stop words analizi (nltk)
#icine hangi dili kullanacagımız parametre alir
stop_word_eng = set(stopwords.words("english"))


# ornek ingilizce metin
text = "There are some examples of handling stop words from some texts."

#burada ön hazırlık prepocesing de yapmamız lazım
# bir text list elde etmek her kelimeyi token laştırmak lazım
text_list = text.split()

filtered_words = [word for word in text_list if word.lower() not in stop_word_eng ]


# %% turkce stop words analizi (nltk)
stop_word_tr = set(stopwords.words("turkish"))
#ornek turkce metin
metin = "merhaba ve arkadaslar cok güzel bir ders işliyoruz. Bu faydalı mı?"
metin_list = metin.split()
filtered_words_tr = [ kelime for kelime in metin_list if kelime.lower() not in stop_word_tr ]

 # mesela yukarı da mı stop word olmasına ragmen islemez
 #cünkü mı? soru eki ile beraber bu stop words listesinde olmadıgı
 #icin onu pas gecer bundan metin on isleme onemlidir
 

# %% kutuphanesiz stop words cikarimi

stopwords_handmade_tr = ["icin", "bu", "ile", "mu", "mi", "özel"]

metin = "Bu bir denemedir. Amacimiz icin bu metinde bulunan özel karekteri el ile girmek elemek mi acaba?"
 
# illa metin list olusturmak zorunda degiliz asagıda ki
# örnekte ki gibi list comprehission icersinde de olusturabilirz

filtered_word_tr_manuel = [ kelime for kelime in metin.split() if kelime.lower() not in stopwords_handmade_tr]

 #stop word tespit etmek istersek ise not in değil in yazarız
stop_word_tr_manuel = [ kelime for kelime in metin.split() if kelime.lower() in stopwords_handmade_tr]
