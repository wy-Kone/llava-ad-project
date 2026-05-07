import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def load_model():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256*28*28,
    max_pixels=512*28*28
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"          
    )
    
    return model, processor

if __name__ == '__main__':
    from PIL import Image
    import os
    import warnings
    warnings.filterwarnings('ignore')
    
    print("正在加载模型...")
    model, processor = load_model()
    print("模型加载成功！")
    
    # 自动找到第一张图片
    image_root = r"E:\学习-模型-数据集\hub\datasets--turing-motors--STRIDE-QA-Bench\snapshots"
    image_path = None
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                image_path = os.path.join(root, f)
                break
        if image_path:
            break
    
    print(f"使用图片: {image_path}")

    image = Image.open(image_path)

    messages = [
        {
            "role":"user",
            "content":[{"type":"image","image":image},{"type":"text","text":"What do you see in this driving scene? Describe any potential hazards."}]
        }
    ]

    #输出处理
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

    #生成回答
    print("正在生成回答...")
    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=200)
    #解码输出
    response = processor.decode(outputs[0],skip_special_tokens=True)
    print(f"模型回答:{response}")