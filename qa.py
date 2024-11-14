from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForQuestionAnswering
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

TEXT_GEN_MODEL_NAME = "gpt2"
QA_MODEL_NAME = "t5-small"  

try:
    text_gen_tokenizer = AutoTokenizer.from_pretrained(TEXT_GEN_MODEL_NAME)
    text_gen_model = AutoModelForCausalLM.from_pretrained(TEXT_GEN_MODEL_NAME)
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

text_gen_pipeline = pipeline("text-generation", model=text_gen_model, tokenizer=text_gen_tokenizer)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

def fetch_context(domain: str) -> str:
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={domain}&format=json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            page = next(iter(data['query']['pages'].values()))
            return page.get('extract', 'No extract found.')
    except requests.RequestException as e:
        print(f"Failed to fetch context from {url}: {e}")
    return "Context not available."

class QuestionRequest(BaseModel):
    question: str
    domain: str

@app.post("/generate")
async def generate_text(data: QuestionRequest):
    question = data.question
    domain = data.domain.replace(" ", "_")  # Replace spaces with underscores for Wikipedia titles

    context = fetch_context(domain)

    input_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

    try:
        generated_text = text_gen_pipeline(input_prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        answer_start_index = generated_text.find("Answer:") + len("Answer:")
        answer = generated_text[answer_start_index:].strip()

        qa_result = qa_pipeline({
            "question": question,
            "context": context + " " + answer  # Combine context and answer for scoring
        })
        score = qa_result['score']

        return {
            "question": question,
            "answer": answer,
            "score": score,
            "context": context[:200]  # Return first 200 characters of context for reference
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("qa:app", host="127.0.0.1", port=8000, reload=True)
