# from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Load pre-trained model (weights)
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# # Set the model in training mode to activate the DropOut modules
# model.train()

# # Define the optimizer
# optimizer = AdamW(model.parameters(), lr=1e-4)

# # Load your training data from a text file
# with open('shakes spehere.txt', 'r') as f:
#     train_data = f.read().splitlines()

# for epoch in range(3):
#     for text in train_data:
#         # Encode the text input
#         indexed_tokens = tokenizer.encode(text, return_tensors='pt')

#         # Clear gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(indexed_tokens)
#         loss = outputs.loss

#         # Backward pass
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# # Save the fine-tuned model
# model.save_pretrained('fine_tuned_model')



# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Load pre-trained model (weights)
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# # Set the device to use
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Set the model in training mode to activate the Dropout modules
# model.train()

# # Define the optimizer
# optimizer = AdamW(model.parameters(), lr=1e-4)

# # Load your training data from a text file
# with open('shakespeare.txt', 'r') as f:
#     train_data = f.read().splitlines()

# for epoch in range(3):
#     for text in train_data:
#         # Encode the text input
#         input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

#         # Clear gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(input_ids=input_ids, labels=input_ids)
#         loss = outputs.loss

#         # Backward pass
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# # Save the fine-tuned model
# model.save_pretrained('fine_tuned_model')


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model in training mode to activate the Dropout modules
model.train()

# Define the optimizer
# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-4, no_deprecation_warning=True)

# Load your training data from a text file
with open('shakespeare.txt', 'r') as f:
    train_data = f.read().splitlines()

for epoch in range(3):
    for i, text in enumerate(train_data):
        # Encode the text input
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

        # Check if input_ids is not empty
        if input_ids.numel() > 0:
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch+1}, Batch: {i+1}/{len(train_data)}, Loss: {loss.item():.4f}")

    print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")


# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')
