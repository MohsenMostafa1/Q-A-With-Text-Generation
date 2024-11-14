# Q-A

### Import necessary libraries

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForQuestionAnswering
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
```
# Load pre-trained models and tokenizers
TEXT_GEN_MODEL_NAME = "gpt2"
QA_MODEL_NAME = "t5-small"
