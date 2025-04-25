import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# Load the model
model_id = "C:/CODING/Hackathon/LLaVA/models/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Load and process the image
image_path = "C:/CODING/Hackathon/harmful_meme.jpg"
image = Image.open(image_path)

# Prepare the prompt
prompt = "<image>\nUSER: This content was flagged as harmful. Please explain why.\nASSISTANT:"

# Process the inputs
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Generate the response
output = model.generate(**inputs, max_new_tokens=200)
response = processor.batch_decode(output, skip_special_tokens=True)[0]

print(response)
