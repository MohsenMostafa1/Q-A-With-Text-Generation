import requests
import json

# Load questions from the JSON file
with open("test_questions.json", "r") as file:
    data = json.load(file)

# API endpoint
url = "http://127.0.0.1:8000/generate"  # Update to /generate if that's what your API uses

# Iterate over each question in the JSON file and send a POST request
for entry in data["questions"]:
    response = requests.post(url, json=entry)
    print(f"Question: {entry['question']}")
    print(f"Response: {response.json()}\n")
