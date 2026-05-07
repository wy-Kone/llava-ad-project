# 基于 Qwen2-VL 的自动驾驶场景理解系统

## 项目简介

本项目基于 Qwen2-VL-2B 多模态大模型，构建了一个自动驾驶场景理解与问答系统。
通过 LoRA 微调提升模型在驾驶场景下的理解能力，并设计了基于 FAISS 的 RAG 模块，
从交通规则知识库中检索相关规则注入 prompt，使模型回答更专业、更合规。

## 技术栈

- **基础模型**：Qwen2-VL-2B-Instruct
- **微调方法**：LoRA（PEFT库，仅训练0.05%参数）
- **知识库**：FAISS 向量数据库 + sentence-transformers
- **数据集**：STRIDE-QA-Bench（自动生成300条VQA训练数据）

## 项目结构
llava_ad_project/
├── model.py              # 模型加载与基础推理
├── generate_qa.py        # 自动生成VQA训练数据
├── finetune.py           # LoRA 微调
├── test_finetune.py      # 微调前后效果对比
├── rag.py                # RAG 知识库构建与推理
├── compare_rag.py        # RAG 对比实验
├── evaluate.py           # 效果评估
├── driving_qa_dataset.json  # 生成的训练数据（300条）
├── traffic_rules_30.txt     # 交通规则知识库（30条）
└── traffic_rules_500.txt    # 交通规则知识库（500条）

## 核心亮点

1. **自动构建训练数据**：用 Qwen2-VL-2B 对 STRIDE-QA-Bench 图片自动生成300条 VQA 问答对，无需人工标注
2. **LoRA 高效微调**：仅训练108万参数（占总参数0.05%），Loss 从0.27降至0.19，场景要素识别数量从4个提升到7个
3. **RAG 自动检索**：用多模态模型自动提取图片场景关键词，再从500条交通规则中检索相关规则，无需人工干预
4. **系统性对比实验**：设计6组对比实验，验证规则库大小和检索方式对RAG效果的影响

## 实验结果

### LoRA 微调效果
| 指标 | 微调前 | 微调后 |
|------|--------|--------|
| 训练Loss | 0.2736 | 0.1893 |
| 场景要素识别数量 | 4个 | 7个 |

### RAG 检索相关性（满分6分）
| 实验组 | 得分 |
|--------|------|
| 30条规则库 + 自然语言检索 | 0/6 |
| 30条规则库 + 自动关键词检索 | 2/6 |
| 500条规则库 + 自然语言检索 | 3/6 |
| 500条规则库 + 自动关键词检索 | 3/6 |

## 环境配置

```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers peft faiss-cpu datasets sentence-transformers accelerate
```

## 运行步骤

```bash
# 1. 生成训练数据
python generate_qa.py

# 2. LoRA 微调
python finetune.py

# 3. 测试微调效果
python test_finetune.py

# 4. RAG 推理
python rag.py

# 5. 评估
python evaluate.py
```