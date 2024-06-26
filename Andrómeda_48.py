import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carga del modelo preentrenado y el tokenizador
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Texto de ejemplo
text = "Hola, ¿cómo estás?"

# Tokenización del texto
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs, labels=inputs['input_ids'])

# Cálculo de la pérdida y optimización
loss = outputs.loss
loss.backward()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
optimizer.step()

# Generación de texto
input_ids = tokenizer.encode("Hola, ¿cómo estás?", return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)