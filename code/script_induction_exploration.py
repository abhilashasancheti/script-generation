# from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# import torch
# import math

# def read_generated_file(path):
#     with open(path, 'r') as f:
#         lines = f.readlines()
#     return lines

# def calculatePerplexity(sentence,model,tokenizer):
#     input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0) 
#     input_ids = input_ids.to('cpu')
#     with torch.no_grad():
#         outputs = model(input_ids, labels=input_ids)
#     loss, logits = outputs[:2]
#     return math.exp(loss)

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
# model.eval()


# # abstract = "Here is a sequence of events that happend when you go to a restaurant: 1. You walk into the restaurant. 2. You take a seat at a table. 3. You pay. 4. You order food." #6.127416613083634

# abstract = "Here is a sequence of events that happen when you eat in a restaurant:  1. You walk into the restaurant. 2. You order the meal. 3. You finish the meal. 4. You take your seat." #8.78272280668399

# print(calculatePerplexity(abstract.strip(), model, tokenizer))

# # abstract = "Here is a sequence of events that happend when you go to a restaurant: 1. You walk into the restaurant. 2. You take a seat at a table. 3. You order food. 4. You pay." #5.787389210421608
# abstract = "Here is a sequence of events that happen when you eat in a restaurant:  1. You walk into the restaurant. 2. You order the meal. 3. You finish the meal. 4. You pay." #9.407343451168737
# print(calculatePerplexity(abstract.strip(), model, tokenizer))
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')
model.eval()
tokenized_text = tokenizer("Hello, my dog is cute", return_tensors="pt")
print(tokenized_text)
input_ids = tokenized_text["input_ids"]
print(input_ids)
with torch.no_grad():
    outputs = model(input_ids)
    prediction_logits = outputs[0]

print(prediction_logits.shape)
word_id = torch.argmax(prediction_logits).item()
predicted_tokens = tokenizer.convert_ids_to_tokens([word_id])
print(predicted_tokens)




# #prediction for mask1
# predicted_index = torch.argmax(predictions[0, mask1]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token) #returns "baseball"


# #prediction for mask2
# predicted_index = torch.argmax(predictions[0, mask2]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token) #returns "actor"


# #prediction for mask3
# predicted_index = torch.argmax(predictions[0, mask3]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token) # returns "."