#!/usr/bin/env python3
import requests, json

BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen2.5-3b-local"

messages = [{"role": "system", "content": "You are a helpful assistant."}]

print(f"Connected to {BASE_URL} | model={MODEL}")
print("Type /exit to quit, /reset to clear history.\n")

while True:
    user = input("> ").strip()
    if not user:
        continue
    if user == "/exit":
        break
    if user == "/reset":
        messages = messages[:1]
        print("(history cleared)")
        continue

    messages.append({"role": "user", "content": user})
    r = requests.post(f"{BASE_URL}/chat/completions", json={
        "model": MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
    }, timeout=300)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]
    print(reply, "\n")
    messages.append({"role": "assistant", "content": reply})
