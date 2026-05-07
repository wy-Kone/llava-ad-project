import json
import torch
import warnings
warnings.filterwarnings('ignore')
from model import load_model
from rag import build_knowledge_base, retrieve_rules, rag_inference

RULES_FILE = "traffic_rules.txt"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

ORIGINAL_QUESTION = "这个驾驶场景中有什么潜在危险？驾驶员应该怎么做？"
OPTIMIZED_QUESTION = "行人 交叉路口 车速 安全距离 潜在危险 驾驶建议"

if __name__ == '__main__':
    with open('driving_qa_dataset.json', encoding='utf-8') as f:
        data = json.load(f)
    test_image = data[0]['image_path']

    print("构建知识库...")
    rules, index, embedder = build_knowledge_base(RULES_FILE, EMBEDDING_MODEL)
    print(f"规则总数：{len(rules)}条\n")

    print("="*50)
    print("组1：500条规则库 + 原始问题 检索结果")
    print("="*50)
    rules1 = retrieve_rules(ORIGINAL_QUESTION, index, rules, embedder)
    for r in rules1:
        print(f"  {r}")

    print("\n" + "="*50)
    print("组2：500条规则库 + 优化问题 检索结果")
    print("="*50)
    rules2 = retrieve_rules(OPTIMIZED_QUESTION, index, rules, embedder)
    for r in rules2:
        print(f"  {r}")

    print("\n加载模型...")
    model, processor = load_model()

    print("\n" + "="*50)
    print("组1：500条规则库 + 原始问题 RAG回答")
    print("="*50)
    response1, _ = rag_inference(model, processor, test_image, ORIGINAL_QUESTION, rules, index, embedder)
    print(response1)

    print("\n" + "="*50)
    print("组2：500条规则库 + 优化问题 RAG回答")
    print("="*50)
    response2, _ = rag_inference(model, processor, test_image, OPTIMIZED_QUESTION, rules, index, embedder)
    print(response2)