import ollama

# 1. This connects directly to the 'llama3.2:3b' you just downloaded
response = ollama.chat(model='llama3.2:3b', messages=[
    {
        'role': 'user',
        'content': 'I just installed you via the terminal. Are you working locally?',
    },
])

# 2. Print the answer to your VS Code console
print("--- BOT RESPONSE ---")
print(response['message']['content'])