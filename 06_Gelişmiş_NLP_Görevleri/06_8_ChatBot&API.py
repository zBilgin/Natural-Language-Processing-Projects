import openai



client = openai.OpenAI(api_key="sk***")

def chat_with_gpt(prompt, history_list):
    messages = [{"role": "user", "content": f"Bu bizim mesajımız:{prompt}. Konuşma geçmişi: {history_list}"}]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    history_list = []

    while True:
        user_input = input("Kullanıcı tarafından girilen mesaj: ")
        
        if user_input.lower() in ["exit", "q"]:
            print("Konuşma tamamlandı.")
            break
        
        history_list.append(user_input)
        response = chat_with_gpt(user_input, history_list)
        print(f"Chatbot: {response}")

