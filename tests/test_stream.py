import requests
import json


response = requests.post(
    "http://127.0.0.1:8000/rag/stream",
    json={
        "question": "what is ali github link?"
    },
    stream=True,
)

print("Status:", response.status_code)

for line in response.iter_lines(
    decode_unicode=True
):

    if line:

        event = json.loads(line)

        print(event)