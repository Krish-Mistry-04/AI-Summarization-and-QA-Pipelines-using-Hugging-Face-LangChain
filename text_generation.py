from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = pipeline("text-generation",
                 model="mistralai/Mistral-7B-Instruct-v0.1",
                 device=0,
                 max_length=256,
                 truncation=True,
                 )

llm = HuggingFacePipeline(pipeline=model)

#Create a prompt template
template = PromptTemplate("Explain {topic} in detail for a {age} year old to understand.")

chain = template | llm
topic = input("Enter a topic: ")
age = input("Enter an age: ")

#Execute the chain
response = chain.invoke({"topic": topic, "age": age})
print(response)