import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define input prompt
prompt = input("input your prompt: ")

# Encode input prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# Generate text with a maximum length of 50 tokens
output = model.generate(input_ids=input_ids, max_length=50, do_sample=True)

# Decode generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
