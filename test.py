import os
from groq import Groq

print("ENV KEY FOUND? ->", os.getenv("GROQ_API_KEY") is not None)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(chat.choices[0].message.content)
