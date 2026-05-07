import json
import torch
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import Qwen2VLForConditionalGeneration,AutoProcessor

#配置
RULES_FILE = "traffic_rules.txt"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3  # 每次检索最相关的3条规则

#构建向量数据库
def build_knowledge_base(rules_file,embedding_model):
    #读取交通规则文件
    with open(rules_file,'r',encoding='utf-8') as f:
        rules = [line.strip() for line in f.readlines() if line.strip()]

    print(f"加载了{len(rules)}条交通规则")

    #用sentence-transformer把每条交通规则转成向量
    print("正在生成规则向量...")
    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(rules,convert_to_numpy=True)

    #建立FAISS索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    print(f"向量数据库构建完成，维度：{dimension}")
    return rules,index,embedder

#检索函数
def retrieve_rules(query,index,rules,embedder,top_k=TOP_K):
    query_vector = embedder.encode([query],convert_to_numpy=True)
    distances,indices = index.search(query_vector.astype(np.float32),top_k)
    retrieved = [rules[i] for i in indices[0]]
    return retrieved

#自动提取场景关键词
def extract_scene_keywords(model,processor,image_path):
    image = Image.open(image_path).convert('RGB')
    messages = [
        {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":"请用关键词描述这个驾驶场景，包括：道路类型、车辆情况、行人情况、交通设施状态等"}]}
    ]
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs = processor(text=text,images=[image],return_tensors='pt',padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=100)

    response = processor.decode(outputs[0],skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

#RAG推理函数
def rag_inference(model,processor,image_path,question,rules,index,embedder):
    #第一步：自动提取场景关键词
    scene_keywords = extract_scene_keywords(model,processor,image_path)
    print(f"场景关键词：{scene_keywords}")

    #第二步：用场景关键词检索相关内容
    relevant_rules = retrieve_rules(scene_keywords,index,rules,embedder)
    rules_text = "\n".join(relevant_rules)

    #第二步：把规则加入问题
    augmented_question = f"""请根据以下交通规则回答问题：
相关交通规则：
{rules_text}
问题：{question}"""
    #第三步：推理
    image = Image.open(image_path).convert('RGB')
    messages = [
        {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":augmented_question}]}
    ]

    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs = processor(
        text = text,
        images = [image],
        return_tensors = "pt",
        padding = True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=256)

    response = processor.decode(outputs[0],skip_special_tokens=True)
    response = response.split("assistant")[-1].strip()
    return response,relevant_rules
    
#RAG主函数
if __name__ == '__main__':
    import json
    from model import load_model

    #构建知识库
    print("构建交通规则知识库...")
    rules,index,embedder = build_knowledge_base(RULES_FILE,EMBEDDING_MODEL)

    #加载模型
    print("加载模型...")
    model,processor = load_model()

    #读取测试图片
    with open('driving_qa_dataset.json',encoding='utf-8') as f:
        data = json.load(f)
    
    test_image = data[0]['image_path']
    question = "这个驾驶场景中有什么潜在危险？驾驶员应该怎么做"

    print(f"\n测试图片:{test_image.split(chr(92))[-1]}")
    print(f"问题：{question}")

    print("\n--- RAG增强回答 ---")
    response,rules_used = rag_inference(
        model,processor,test_image,question,rules,index,embedder
    )
    print("\n检索到的相关规则:")
    for rule in rules_used:
        print(f"{rule}")
    print(f"\n模型回答:\n{response}")