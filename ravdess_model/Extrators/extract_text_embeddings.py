import os
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

statements = {
    '01': "Kids are talking by the door",
    '02': "Dogs are sitting by the door"
}

statements_embeddings ={}

for stat_id, text in statements.items():
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddigs = outputs.last_hidden_state[:, 0, :].squeeze()
    statements_embeddings[stat_id] = embeddigs
    print(f"Statement {stat_id} embeddings shape: {embeddigs.shape}")

    torch.save(statements_embeddings, "text_embeddings.pt")
    print("Saved!")