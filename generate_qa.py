import os
import json
import torch
from PIL import Image
from model import load_model
QUESTIONS = [
    "What objects do you see in this driving scene?",
    "What are the potential hazards in this scene?",
    "What should the driver do in this situation?"
]

IMAGE_ROOT = r"E:\学习-模型-数据集\hub\datasets--turing-motors--STRIDE-QA-Bench\snapshots"
OUTPUT_FILE = "driving_qa_dataset.json"
MAX_IMAGES = 100

#推理函数
def generate_answer(model,processor,image,question):
    messages = [
        {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]}
    ]

    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs = processor(text=text,images=[image],return_tensors="pt",padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=512)
    
    response = processor.decode(outputs[0],skip_special_tokens=True)
    #只取assistant的回答部分
    response = response.split("assistant")[-1].strip()
    return response

#主函数
def main():
    print("正在加在模型...")
    model,processor = load_model()
    print("模型加载成功！")
    
    #收集所有图片路径
    image_paths = []
    for root,dirs,files in os.walk(IMAGE_ROOT):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                image_paths.append(os.path.join(root,f))
                if len(image_paths) >= MAX_IMAGES: 
                    break
        if len(image_paths) >=MAX_IMAGES:
            break
    
    print(f"找到{len(image_paths)}张图片，开始生成问答对：")

    dataset = []
    for i,image_path in enumerate(image_paths):
        print(f"处理第{i+1}/{len(image_paths)}张图片...")
        image = Image.open(image_path)

        for question in QUESTIONS:
            answer = generate_answer(model,processor,image,question)
            dataset.append(
                {'image_path':image_path,'question':question,'answer':answer}
            )
    
    #保存成JSON文件
    with open(OUTPUT_FILE,'w',encoding='utf-8') as f:
        json.dump(dataset,f,ensure_ascii=False,indent=2)

    print(f"完成！共生成{len(dataset)}条问答对，保存到{OUTPUT_FILE}")

if __name__ == '__main__':
    main()