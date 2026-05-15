import os
os.environ['HF_HOME'] = r'E:\学习-模型-数据集\llava_ad_project'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'E:\学习-模型-数据集\hub'
import torch
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

#配置
PDF_FILE = "中华人民共和国道路交通安全法.pdf"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#pdf加载和分块函数
def build_knowledge_base_from_pdf(pdf_file,embedding_model):
    print(f"加载PDF：{pdf_file}")

    #用langchain加载PDF
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    print(f"PDF共{len(documents)}页")

    #用RecursiveCharacterTextSplitter分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n","\n","。",",",""]
    )
    chunks = splitter.split_documents(documents)
    print(f"分块完成，共{len(chunks)}个chunk")

    #提取文本内容
    texts = [chunk.page_content for chunk in chunks]

    #用 sentence—transformers 转成向量
    print("正在生成向量...")
    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(texts,convert_to_numpy=True)

    #建立FAISS索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    print(f"知识库构建完成，维度：{dimension},chunk数量：{len(texts)}")
    return texts,index,embedder

#构建检索函数和RAG推理函数
def retrieve_chunks(query,index,texts,embedder,top_k=TOP_K):
    """检索最相关的 chunk """
    query_vector = embedder.encode([query],convert_to_numpy=True)
    distances,indices = index.search(query_vector.astype(np.float32),top_k)
    retrieved = [texts[i] for i in indices[0]]
    return retrieved

def extract_scene_keywords(model,processor,image_path):
    """自动提取场景关键词"""
    image = Image.open(image_path).convert('RGB')
    messages =[
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "请用关键词描述这个驾驶场景，包括：道路类型、车辆情况、行人情况、交通设施状态等"}
        ]}
    ]
    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs = processor(text=text,images=[image],return_tensors='pt',padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=100)

    response = processor.decode(outputs[0],skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

def rag_inference_pdf(model, processor, image_path, question, texts, index, embedder):
    """基于 PDF 知识库的 RAG 推理"""
    # 第一步：自动提取场景关键词
    scene_keywords = extract_scene_keywords(model, processor, image_path)
    print(f"场景关键词：{scene_keywords}")
    
    # 第二步：用关键词检索相关 chunk
    relevant_chunks = retrieve_chunks(scene_keywords, index, texts, embedder)
    chunks_text = "\n\n".join(relevant_chunks)
    
    # 第三步：构建增强 prompt
    augmented_question = f"""请根据以下交通法规内容回答问题：

相关法规：
{chunks_text}

问题：{question}"""
    
    # 第四步：推理
    image = Image.open(image_path).convert('RGB')
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": augmented_question}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors='pt', padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    response = response.split("assistant")[-1].strip()
    return response, relevant_chunks

#主函数
if __name__ == '__main__':
    import json
    from model import load_model

    # 构建 PDF 知识库
    print("构建 PDF 知识库...")
    texts, index, embedder = build_knowledge_base_from_pdf(PDF_FILE, EMBEDDING_MODEL)

    # 加载模型
    print("加载模型...")
    model, processor = load_model()

    # 读取测试图片
    with open('driving_qa_dataset.json', encoding='utf-8') as f:
        data = json.load(f)

    test_image = data[0]['image_path'].replace(
    r'E:\学习-模型-数据集\hub',
    r'E:\学习-模型-数据集\llava_ad_project\hub'
    )
    question = "这个驾驶场景中有什么潜在危险？驾驶员应该怎么做？"

    print(f"\n测试图片：{test_image.split(chr(92))[-1]}")
    print(f"问题：{question}")

    print("\n--- PDF RAG 增强回答 ---")
    response, chunks_used = rag_inference_pdf(
        model, processor, test_image, question, texts, index, embedder
    )
    print("\n检索到的相关法规片段：")
    for i, chunk in enumerate(chunks_used):
        print(f"\n片段{i+1}：{chunk[:100]}...")
    print(f"\n模型回答：\n{response}")