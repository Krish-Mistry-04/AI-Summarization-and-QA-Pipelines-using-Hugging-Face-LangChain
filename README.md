# AI Summarization and QA Pipelines using Hugging Face & LangChain

This project demonstrates the use of Hugging Face Transformers integrated with LangChain to create powerful NLP pipelines for text summarization, prompt-based text generation, and question-answering.

## 🔍 Features

- 📄 **Text Summarization:** Uses `facebook/bart-large-cnn` for efficient summarization of long-form text.
- 💬 **Prompt-based Generation:** Leverages `mistralai/Mistral-7B-Instruct-v0.1` to explain any topic at an age-specific comprehension level.
- ❓ **Question Answering (QA):** Answers user-generated questions based on generated summaries using `deepset/roberta-base-squad2`.

## 📁 Project Structure


├── main.py # Basic summarization with Hugging Face pipeline

├── main2.py # Topic-based generation using LangChain & Mistral

├── main3.py # Combined summarization + refinement + QA system

## 🚀 Requirements

- Python 3.10
- torch>=2.6.0 (for secure model loading)
- transformers
- langchain
- langchain-huggingface
- safetensors

```bash
pip install torch transformers langchain langchain-huggingface safetensors
```
## ✨Example Use Case
- Input a news article or research paper → get a summary → ask questions about it interactively.

