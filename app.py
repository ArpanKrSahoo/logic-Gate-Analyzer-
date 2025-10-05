import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


MODEL_DIR = 'logic_gate_model' # replace with path where you saved model


@st.cache_resource
def load_resources():
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
return tokenizer, model, device


tokenizer, model, device = load_resources()


st.title('Logic Gate Identifier')
st.write('Type a natural language description of a logic behavior; model predicts the gate.')


text = st.text_area('Description', value='The output is high only when both inputs are high.')


if st.button('Predict'):
if not text.strip():
st.warning('Enter some text first')
else:
enc = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors='pt')
enc = {k:v.to(device) for k,v in enc.items()}
model.eval()
with torch.no_grad():
logits = model(**enc).logits
probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
pred = np.argmax(probs)
# label order depends on training label encoder. Display all scores.
st.subheader(f'Predicted: {pred}')
st.write('Probabilities (raw indices):')
st.write(probs)
st.write('Note: map index -> label using the training label encoder.\n'
'If you used LabelEncoder in training, save label encoder classes to a JSON or text file and load here to show human labels.')