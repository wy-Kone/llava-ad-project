import json
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATA_FILE = "driving_qa_dataset.json"
OUTPUT_DIR = "lora_finetuned"
BATCH_SIZE = 1
NUM_EPOCHS = 3
LEARNING_RATE = 2E-4
MAX_LENGTH =1024

class DrivingQADataset(Dataset):
    def __init__ (self,data_file,processor,max_length=512):
        with open(data_file,'r',encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        question = item['question']
        answer = item['answer']

        messages = [
            {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]},{"role":"assistant","content":answer}
        ]

        text = self.processor.apply_chat_template(
            messages,tokenize=False,add_generation_prompt=False
        )
        inputs = self.processor(
            text = text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # 只对回答部分计算损失
        input_ids = inputs['input_ids']
        labels = input_ids.clone()

        # 找到 assistant 开始的位置，之前的全部设为 -100
        text_ids = self.processor.tokenizer.encode("assistant")
        for i in range(len(labels) - len(text_ids)):
            if labels[i:i+len(text_ids)].tolist() == text_ids:
                labels[:i+len(text_ids)] = -100
                break

        inputs['labels'] = labels
        return inputs
    
def load_model_for_training():
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

    #给模型加上LoRA
    lora_config = LoraConfig(
        r = 8,
        lora_alpha=16,  #缩放系数，通常是秩的2倍
        target_modules=["q_proj","v_proj"],   #给哪些层加上LoRA
        lora_dropout=0.1,  #防止过拟合
        bias="none",
        task_type="CAUSAL_LM"  #因果语言模型
    )

    model = get_peft_model(model,lora_config)
    model.print_trainable_parameters()  #打印可训练参数量

    return model,processor
    
def train():
    print("加载模型...")
    model,processor = load_model_for_training()
        
    print("加载数据集...")
    dataset = DrivingQADataset(DATA_FILE,processor,MAX_LENGTH)
    dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for step,batch in enumerate(dataloader):
            batch = {k:v.to(model.device) for k,v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch{epoch+1}/{NUM_EPOCHS},Step{step},Loss:{loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch{epoch+1} 平均Loss:{avg_loss:.4f}")
    
    print("保存模型...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"模型保存到{OUTPUT_DIR}")
if __name__ == '__main__':
    train()