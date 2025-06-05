# metinlerde bulunan fazla bosluklari ortadan kaldir
text = "Hello,     World!       2035"

"""
text.split()
Out[2]: ['Hello,', 'World!', '2035']
"""

cleaned_text1 = " ".join(text.split())
print(f"text: {text} \n cleaned_text1: {cleaned_text1}")


# %% buyuk -> kucuk harf cevrimi
text = "Hello, World! 2035"
cleaned_text2 = text.lower() # kucuk harfe cevir
print(f"text: {text} \n cleaned_text2: {cleaned_text2}")

# %% noktalama isaretlerini kaldir
import string 

text = "Hello, World! 2035"

cleaned_text3 = text.translate(str.maketrans("", "", string.punctuation))
print(f"text: {text} \n cleaned_text3: {cleaned_text3}")



# %% ozel karakterleri kaldir, %, @, /,*, #
import re 

text = "Hello, World! 2035%"

cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]", "", text)
print(f"text: {text} \n cleaned_text4: {cleaned_text4}")


# %% yazim hatalarini duzelt
from textblob import TextBlob # metin analizlerinde kullanilan bir kutuphane

text = "Hellıo, Wirld! 2035"
cleaned_text5 = TextBlob(text).correct() # correct: yazim hatalarini duzeltir
print(f"text: {text} \n cleaned_text5: {cleaned_text5}")

# %% html yada url etiketlerini kaldir
from bs4 import BeautifulSoup

html_text = "<div>Hello, World! 2035</div>" # html etiketi var
# beautiful soup ile html yapisini parse et, get_text ile text kismini cek
cleaned_text6 = BeautifulSoup(html_text, "html.parser").get_text()
print(f"text: {html_text} \n cleaned_text6: {cleaned_text6}")


















