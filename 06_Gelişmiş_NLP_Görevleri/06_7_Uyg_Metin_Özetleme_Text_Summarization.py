from transformers import pipeline

# ozetleme pipeline yukle
summarizer = pipeline("summarization")

text = """
Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use 
to progressively improve their performance on a specific task. Machine learning algorithms build a mathematical 
model of sample data, known as "training data", in order to make predictions or decisions without being explicitly 
programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, 
detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific 
instructions for performing the task. Machine learning is closely related to computational statistics, which focuses 
on making predictions using computers. The study of mathematical optimization delivers methods, theory and application 
domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory 
data analysis through unsupervised learning. In its application across business problems, machine learning is also referred 
to as predictive analytics.
"""

# metin ozetleme
summary = summarizer(
    text,
    max_length = 20,
    min_length  = 5,
    do_sample = True
    )

# do_sample=False: daha tutarlı ve belirgin çıktılar (greedy decoding gibi)
# do_sample=True: daha yaratıcı, çeşitli ifadeler (sampling, top-k/top-p sampling yapılabilir)
# max_length çok düşük seçilirse anlam eksikliği oluşabilir.



print(summary[0]["summary_text"])

# %% Haber Metni Özeti

summarizer = pipeline("summarization")

news_text = """
The European Union announced a new set of regulations today aimed at curbing the environmental impact of the fashion industry. 
The new policy encourages companies to implement more sustainable practices, such as using recyclable materials and reducing waste 
in production processes. Major fashion brands have welcomed the regulations, stating they align with their own environmental goals.
"""

summary = summarizer(news_text, max_length=30, min_length=10, do_sample=False)
print(summary[0]["summary_text"])


# %% Bilimsel Makale Özeti

science_text = """
Quantum computing is an area of computing focused on developing computer technology based on the principles of quantum theory. 
Quantum computers use quantum bits, or qubits, which can be in superpositions of states. This allows them to perform certain 
types of computation much faster than classical computers. Although quantum computing is still in its early stages, it holds 
great promise for solving problems in cryptography, chemistry, and complex simulations.
"""

summary = summarizer(science_text, max_length=40, min_length=15, do_sample=False)
print(summary[0]["summary_text"])

# %% Blog Yazısı Özeti
blog_text = """
If you're just starting out in programming, it's important to build a strong foundation. Begin with learning basic syntax and 
control structures in a beginner-friendly language like Python. Then, move on to understanding algorithms, data structures, 
and problem-solving strategies. Building small projects is a great way to practice. Don't be afraid to make mistakes — every 
bug is a learning opportunity!
"""

summary = summarizer(blog_text, max_length=35, min_length=10, do_sample=True)
print(summary[0]["summary_text"])


