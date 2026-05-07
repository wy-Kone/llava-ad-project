import torch
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LORA_DIR = "lora_finetuned"

#模型加载函数
def load_original_model():
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256*28*28,
        max_pixels=512*28*28
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model,processor

def load_finetuned_model():
    processor = AutoProcessor.from_pretrained(
        LORA_DIR,
        min_pixels=256*28*28,
        max_pixels=512*28*28
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, LORA_DIR)
    return model, processor

#推理函数和主函数
def generate_response(model,processor,image_path,question):
    image = Image.open(image_path).convert('RGB')

    messages = [
        {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]}
    ]

    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs = processor(
        text = text,
        images =[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=256)

    response = processor.decode(outputs[0],skip_special_tokens=True)
    response = response.split("assistant")[-1].strip()
    return response

if __name__ == '__main__':
    import json
    with open('driving_qa_dataset.json',encoding='utf-8') as f:
        data = json.load(f)

    test_image = data[0]['image_path']
    question = "What are the potential hazards in this scene?"

    print("="*50)
    print("测试图片：",test_image.split("\\")[-1])
    print("问题：",question)

    print("\n--- 原始模型回答 ---")
    model,processor = load_original_model()
    response = generate_response(model,processor,test_image,question)
    print(response)
    del model
    torch.cuda.empty_cache()

    print("\n--- 微调后模型回答 ---")
    model,processor = load_finetuned_model()
    response = generate_response(model,processor,test_image,question)
    print(response)