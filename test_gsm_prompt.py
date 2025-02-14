import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

peft_model_id = sys.argv[1] if len(sys.argv) > 1 else "./gsm8k-lora"

PROMPT = "Q: {question}\nA: "

def format_prompt(**kwargs):
    return PROMPT.format(**kwargs)

try:
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
    model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
except:
    model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

while True:
    question = input("> ")
    question = format_prompt(question=question)
    inputs = tokenizer(question, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, min_new_tokens=1)
        print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))