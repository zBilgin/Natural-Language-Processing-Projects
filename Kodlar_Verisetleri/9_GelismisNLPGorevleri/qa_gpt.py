from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

import warnings
warnings.filterwarnings("ignore")

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_answer(context, question):
    
    input_text = f"Question: {question}, Context: {context}. Please answer the question according to context"
    
    # tokenize
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # modeli calistir
    with torch.no_grad():
        outputs = model.generate(inputs, max_length = 300)
        
    # uretilen yaniti decode edelim
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True) # merhaba<EOS><PAD>
    
    # tanitlari ayiklayalim
    answer = answer.split("Answer:")[-1].strip()
    
    return answer

question = "What is the capital of France"
context = "France, officially the French Republic, is a country whose capital is Paris"

answer = generate_answer(context, question)
print(f"Answer: {answer}")

"""
Answer: The capital of France is Paris.
"""

question = '''What is Machine Learning?'''
context = ''' Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance 
                on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or 
                decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection 
                of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning 
                is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, 
                theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory 
                data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics. '''


answer = generate_answer(context, question)
print(f"Answer: {answer}")



















    
    
    