from transformers import BertTokenizer
from transformers import BertModel
tokenizer=BertTokenizer.from_pretrained(
    "bert-base-uncased"
)
model=BertModel.from_pretrained(
    "bert-base-uncased"
)
text="I love machine learning"
tokens=tokenizer.tokenize(text)
print(tokens)
encoded=tokenizer.encode(text)
print(encoded)
from transformers import BertTokenizer
from transformers import BertModel
import torch
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)
model = BertModel.from_pretrained(
    "bert-base-uncased"
)
text = "I love machine learning"
inputs = tokenizer(
    text,
    return_tensors="pt"
)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
