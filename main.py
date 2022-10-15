import base64
import io
import urllib
from typing import List
import torch
from fastapi import FastAPI
from pydantic import BaseModel

class Input(BaseModel):
    numbers: List[float]

app = FastAPI()

encoded_model = urllib.request.urlopen("https://gist.githubusercontent.com/WKuipers/a230437dcc2f3af050955b272853a392/raw/a6e43b6f150153f279b9c0c7971dcb065a98be24/model_scripted.pth").read()
with io.BytesIO(base64.b64decode(encoded_model)) as f:
    model = torch.jit.load(f)
    f.seek(0)
    content = f.read()
model.eval()
with open("model_scripted.pt", "wb") as wf:
    wf.write(content)

@app.post("/")
def serve_model(data: Input):
    tensor = torch.tensor(data.numbers)
    return model.forward(tensor).tolist()
