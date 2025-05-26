import http.client
import json

def query_llm(prompt):
    conn = http.client.HTTPSConnection("api.chatanywhere.tech")
    payload_dict = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    payload = json.dumps(payload_dict)

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer sk-YOUKEY'
    }
    conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
    res = conn.getresponse()
    data = res.read()
    response = json.loads(data)
    message = response["choices"][0]["message"]["content"]
    return message.strip('```python').strip('```')

# query_llm("What is the capital of France?")  