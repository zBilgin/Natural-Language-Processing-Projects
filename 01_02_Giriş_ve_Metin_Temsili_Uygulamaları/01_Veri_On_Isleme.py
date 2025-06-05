# metinlerde bulunan fazla boşlukları ortadan kaldir

text = "Hello,       World!   2037"

"""
text.split()
Out[6]: ['Hello,', 'World!', '2037']
"""

#Split texti ayırır, kelime öbeği olarak 
#Join ise ayrılanı birleştirir böylelikle boşluklardan
#kurtulmuş olur kanımca str replace ile de bu yapılabilir

#joinin baş parametresi nasıl birleştirecğini alır
# "" şeklinde olursa boşluksuz birleştirir
# "wow" olursa arasında wow ile birleştirir 

cleaned_text1 = " ".join(text.split())
print("cleaned_text1:", cleaned_text1)

# %% Buyuk harf kucuk harf cevrimi

text = "Hello, World! 2037"
cleaned_text2 = text.lower() #kucuk harfe cevirir
print(f" text: {text}\n cleaned_text2: {cleaned_text2}")

# %% Noktalama isareti kaldir
import string
#string modülü tüm noktalama işaretlerini içeren bir sabit sağlar bize

text = "Hello, World! 2035"

#translate fonksiyonu kkulanağız string fonk ile de noktalama işaretini bir şey ile değiştirmeden ortadan kaldıracağız
# string.punctuation → tüm noktalama karakterlerini içeren string
# str.translate() → karakterleri eşleştirip silmemizi sağlar
cleaned_text3 = text.translate(str.maketrans("","",string.punctuation))
print(f" text: {text}\n cleaned_text3: {cleaned_text3}")


# %% Ozel karekterleri kaldir  #,$,@,*,/ gibi 

import re 
# re => regular expression kütüphanesi regex ?
# re modulü düzenli ifadeler ile çalışmayı sağlar veri temizlemede oldukça sık kulanılır

text = "Hello, World! 2035% #,$,@,*,/" 

# Sadece harf, rakam ve boşlukları bırak, diğerlerini sil
cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]","", text)
print(f" text: {text}\n cleaned_text4: {cleaned_text4}")

# %% Yazim hatalarini kaldir

from textblob import TextBlob 
#Metin analizlerin de kullanılan bir kutuphane

text = "Hellıo,  Wirld! 2037 How are yao?"
cleaned_text5 = TextBlob(text).correct() 
#correct: yazım hatalarini düzeltir 
#bu çok güçlü bir kütüphane değil basit yazım hatalarını düzeltir
# correct(): basit yazım hatalarını düzeltir (AI tabanlı değil, kısıtlı)

print(f" text: {text}\n cleaned_text5: {cleaned_text5}")


# %% html yada url etiketlerini kaldir  
from bs4 import BeautifulSoup

html_text = "<div>Hello, World! 2035</div>" 

#beautiful soup ile html kısmını parse et, get text ile text kısmını çek
# HTML etiketlerini sil, yalnızca düz metni al
cleaned_text6 = BeautifulSoup(html_text, "html.parser").getText()

print(f" html_text: {html_text}\n cleaned_text6: {cleaned_text6}")