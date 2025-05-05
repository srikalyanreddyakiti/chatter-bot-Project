from transformers import AutoModelWithLMHead, AutoTokenizer

# Load pre-trained GPT-2 model and tokenizer
model = AutoModelWithLMHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Get input from user
prompt = "What is the meaning of life?"

# Tokenize input
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate output
outputs = model.generate(inputs, max_length=100, do_sample=True)

# Decode output and print to console
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
