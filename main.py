
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = FastAPI()

class Text(BaseModel):
    text: str

# Load model and tokenizer
MODEL_NAME = "pranay5255/assignment_orbitshift"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def get_prediction(text: str):
    class_dict = {0: 'CLASS1',
                1: 'CLASS10',
                2: 'CLASS11',
                3: 'CLASS12',
                4: 'CLASS13',
                5: 'CLASS2',
                6: 'CLASS3',
                7: 'CLASS4',
                8: 'CLASS5',
                9: 'CLASS6',
                10: 'CLASS7',
                11: 'CLASS8',
                12: 'CLASS9'}

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    predicts = torch.nn.functional.softmax(outputs.logits, dim =-1)
    # executing argmax function to get the candidate label
    return class_dict[np.argmax(predicts.cpu().detach().numpy(),axis=1)[0]]

@app.get("/")
def read_root():
    return {"message": "Hello, this is a text classification API using the model pranay5255/assignment_orbitshift."}

@app.post("/predict")
def predict(text_input: Text):
    try:
        result = get_prediction(text_input.text)
        return {"Predicted Class of Title": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

