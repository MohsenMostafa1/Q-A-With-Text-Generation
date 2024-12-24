from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import aiohttp
import asyncio
from cachetools import TTLCache
import torch

app = FastAPI()

# Load improved QA model
qa_model_name = "deepset/roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load a better sentence retriever
retriever_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Cache for Wikipedia content
context_cache = TTLCache(maxsize=100, ttl=3600)

class QuestionRequest(BaseModel):
    question: str
    domain: str

async def fetch_wikipedia_context(domain: str) -> str:
    """Fetch Wikipedia context."""
    if domain in context_cache:
        return context_cache[domain]

    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={domain}&format=json"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    page = next(iter(data['query']['pages'].values()))
                    extract = page.get('extract', 'No extract found.')
                    context_cache[domain] = extract  # Cache the result
                    return extract
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout fetching Wikipedia context.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch context: {e}")

    return "Context not available."

def retrieve_relevant_context(question: str, context: str, top_k: int = 5) -> str:
    """Retrieve top-k relevant sentences from the context using a retriever."""
    sentences = context.split(".")  # Split into sentences
    question_embedding = retriever_model.encode(question, convert_to_tensor=True)
    sentence_embeddings = retriever_model.encode(sentences, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)
    top_indices = torch.topk(similarities[0], k=min(top_k, len(sentences))).indices

    # Collect top-k sentences
    top_sentences = [sentences[i].strip() for i in top_indices if sentences[i].strip()]
    return ". ".join(top_sentences) + "."

@app.post("/generate")
async def generate_answer(data: QuestionRequest):
    question = data.question.strip()
    domain = data.domain.strip().replace(" ", "_")  # Replace spaces with underscores for Wikipedia titles

    if not question or not domain:
        raise HTTPException(status_code=400, detail="Invalid question or domain.")

    # Fetch context asynchronously
    context = await fetch_wikipedia_context(domain)

    if context == "No extract found." or not context:
        raise HTTPException(status_code=404, detail="Context not found.")

    # Retrieve relevant sentences
    relevant_context = retrieve_relevant_context(question, context)

    # Answer question asynchronously
    loop = asyncio.get_event_loop()
    qa_result = await loop.run_in_executor(None, qa_pipeline, {
        "question": question,
        "context": relevant_context
    })

    return {
        "question": question,
        "answer": qa_result.get("answer", "No answer found."),
        "score": qa_result.get("score", 0),
        "relevant_context": relevant_context,
        "qa_model": qa_model_name
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("QuestionSS:app", host="127.0.0.1", port=8000, reload=True)
