import openai

openai.api_key = "api_key"

def chat_with_gpt(prompt, history_list):
    
    responce = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role":"user", "content": f"Bu bizim mesajımız: {prompt}. Konuşma geçmişi: {history_list}"}]
        )
    
    return responce.choices[0].message.content.strip()

if __name__ == "__main__":
    
    history_list = []
    
    while True:
        
        user_input = input("Kullanıcı tarafından girilen mesaj:" )
        
        if user_input.lower() in ["exit", "q"]:
            print("Konuşma tamamlandi")
            break
        history_list.append(user_input)
        responce = chat_with_gpt(user_input, history_list)
        print(f"Chatbot: {responce}")
    
    
# https://platform.openai.com/api-keys    
    
    
    
    
    
    
    
    
    