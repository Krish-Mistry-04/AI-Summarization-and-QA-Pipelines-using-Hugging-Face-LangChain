import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import pipeline
import torch

# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Summarization pipeline
model = pipeline("summarization", model="facebook/bart-large-cnn")
response = model("In recent years, the rapid advancement of artificial intelligence (AI) has transformed various industries, including healthcare, finance, and education. AI-powered tools are now being used to detect diseases, automate financial trading, and personalize learning experiences. However, these innovations have also raised significant ethical concerns. Issues such as data privacy, algorithmic bias, and the displacement of human labor have sparked global debates. Policymakers, researchers, and technologists are actively working together to create guidelines that promote the responsible use of AI. As the technology continues to evolve, it becomes increasingly important to ensure that its development aligns with societal values and benefits all segments of the population.")
print(response)
