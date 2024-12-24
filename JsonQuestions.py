import aiohttp
import asyncio
import json

async def send_request(session, url, payload):
    async with session.post(url, json=payload) as response:
        return await response.json()

async def process_questions(file_path, api_url):
    with open(file_path, "r") as file:
        data = json.load(file)

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, api_url, entry) for entry in data["questions"]]
        responses = await asyncio.gather(*tasks)

    for question, response in zip(data["questions"], responses):
        print(f"Question: {question['question']}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    asyncio.run(process_questions("test_questions.json", "http://127.0.0.1:8000/generate"))
