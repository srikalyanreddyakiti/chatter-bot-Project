import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the Dropout modules
model.eval()

# Encode a text input
text = "Once upon a time"
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens to a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Generate text
with torch.no_grad():
    outputs = model.generate(tokens_tensor)
    prediction = outputs[0].tolist()

# Decode the prediction
predicted_text = tokenizer.decode(prediction, skip_special_tokens=True)

print(predicted_text)
