from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

refinement_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

summarization_chain = summary_template | summarizer | refiner

text_to_summarize = input("Enter text to summarize: ")
length = input("Enter summary length (short/medium/long): ")
length_map = {"short": 50, "medium": 150, "long": 300}
max_length = length_map.get(length.lower(), 150)

summary = summarization_chain.invoke({"text": text_to_summarize, "length": max_length})
print("\n **Generated Summary:**")
print(summary)

while True:
    question = input("\nEnter a question about the text (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    qa_result = qa_pipeline(question=question, context=summary)
    print("\n**Answer:**")
    print(qa_result['answer'])