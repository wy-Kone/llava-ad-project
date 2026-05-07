# 六组实验的检索规则（第1、2组没有RAG，不需要打分）
RAG_RESULTS = {
    "第3组：30条+自然语言": [
        "驾驶员不得酒后驾驶、疲劳驾驶或分心驾驶",
        "恶劣天气或传感器受遮挡时，自动驾驶系统应及时提示驾驶员接管车辆",
        "自动驾驶车辆在系统故障时，应立即降低速度"
    ],
    "第4组：500条+自然语言": [
        "驾驶员应认识到行人行为具有不确定性，应采用防御性驾驶策略",
        "自动驾驶车辆遇前车急刹风险时，应提前减速",
        "自动驾驶车辆安全停车时，应开启危险报警闪光灯"
    ],
    "第5组：30条+自动关键词": [
        "车辆应按照道路交通标志、标线和交通信号灯的指示行驶",
        "车辆在狭窄道路会车时，应减速靠右行驶",
        "自动驾驶系统在道路施工场景下，应识别异常道路环境"
    ],
    "第6组：500条+自动关键词": [
        "车辆在非机动车道旁行驶时，应注意行人可能借道通行",
        "车辆在双向多车道人行横道前，应注意相邻车道车辆遮挡行人",
        "车辆通过乡村道路时，应注意行人可能在车行道内行走"
    ]
}

# LoRA 训练 Loss 记录
LOSS_HISTORY = {
    "Epoch1": 0.2736,
    "Epoch2": 0.2150,
    "Epoch3": 0.1893
}

# 微调前后场景要素识别
ORIGINAL_ELEMENTS = ["pedestrians", "traffic lights", "vehicles", "road conditions"]
FINETUNED_ELEMENTS = ["pedestrians", "vehicles", "bus", "traffic lights", "road conditions", "weather", "crosswalks"]

#LoRA微调效果评估
def evaluate_finetune():
    print("\n" + "=" * 50)
    print("LoRA微调效果评估")
    print("=" * 50)

    print("\n【Loss下降曲线】")
    for epoch,loss in LOSS_HISTORY.items():
        bar = "█" * int(loss * 20)
        print(f"{epoch}:{bar} {loss:.4f}")
    
    print("\n【场景要素识别对比】")
    print(f"原始模型识别要素数量：{len(ORIGINAL_ELEMENTS)}")
    for e in ORIGINAL_ELEMENTS:
        print(f"  ✓ {e}")
    
    print(f"\n微调后模型识别要素数量：{len(FINETUNED_ELEMENTS)}")
    for e in FINETUNED_ELEMENTS:
        print(f"  ✓ {e}")
    
    improvement = len(FINETUNED_ELEMENTS) - len(ORIGINAL_ELEMENTS)
    print(f"\n提升：多识别出 {improvement} 个场景要素")

def evaluate_rag():
    print("=" * 50)
    print("RAG检索相关性评估（人工打分）")
    print("=" * 50)
    print("评分标准：0=完全不相关，1=稍相关，2=非常相关\n")

    SCORES = {
        "第3组：30条+自然语言":    [0, 0, 0],
        "第4组：500条+自然语言":   [2, 1, 0],
        "第5组：30条+自动关键词":  [2, 0, 0],
        "第6组：500条+自动关键词": [1, 2, 0],
    }

    RULES = list(RAG_RESULTS.values())

    print("各组详细得分：")
    for i, (group_name, rule_scores) in enumerate(SCORES.items()):
        print(f"\n{group_name}:")
        rules = RULES[i]
        for rule, score in zip(rules, rule_scores):
            print(f"  规则：{rule[:30]}... 得分：{score}/2")
        total = sum(rule_scores)
        bar = "█" * total + "░" * (6 - total)
        print(f"  本组总分：{bar} {total}/6")

    print("\n" + "=" * 50)
    print("RAG各组得分汇总：")
    for group_name, rule_scores in SCORES.items():
        total = sum(rule_scores)
        bar = "█" * total + "░" * (6 - total)
        print(f"{group_name}: {bar} {total}/6")

if __name__ == '__main__':
    evaluate_finetune()
    evaluate_rag()