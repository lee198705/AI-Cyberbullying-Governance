# AI Cyberbullying Governance

一个根据《互联网＋作品》技术路线重新工程化的可运行项目，核心覆盖评论监测、恶意评论识别、风险分级、用户画像、高赞理性评论排序和个性化内容推荐。

运行环境要求：Python >= 3.10。

## 功能模块

- 评论监测：读取评论流，输出攻击性概率、风险等级和处理动作。
- 恶意评论治理：按 0.35 / 0.60 / 0.80 阈值完成提示、限流、阻断和人工审核分流。
- 用户画像：结合历史评论、互动行为、情绪、攻击风险、表达理性度和话题兴趣生成画像。
- 话题级 A1 / B / A2：用 `src/stance.py` 单独计算观点倾向，同一用户在不同话题下保留不同标签，不用情绪分类代替立场判断。
- 理性评论排序：综合点赞、理性程度、中立性、质量和攻击风险排序。
- 内容推荐：融合兴趣匹配、内容质量、多样性和文章内容风险约束生成 Top-N 推荐，同时展示评论区讨论风险。
- BiLSTM 训练入口：毒性识别和情感分类均支持训练 checkpoint，并在无模型环境下自动回退到轻量规则。

## 运行

```bash
cd AI-Cyberbullying-Governance
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

如果只想验证核心逻辑：

```bash
python3 -m unittest discover -s tests
```

训练 COLDataset 毒性识别模型：

```bash
python3 -m pip install -r requirements-train.txt
python3 scripts/train_toxicity.py \
  --train ../COLDataset-main/COLDataset/train.csv \
  --dev ../COLDataset-main/COLDataset/dev.csv
```

训练三分类情感模型（CSV 列为 `TEXT,label`，标签支持 `negative/neutral/positive` 或 `0/1/2`）：

```bash
python3 scripts/train_sentiment.py \
  --train path/to/sentiment_train.csv \
  --dev path/to/sentiment_dev.csv
```

## 项目结构

```text
AI-Cyberbullying-Governance/
├── app.py
├── data/
├── models/
├── scripts/
├── src/
└── tests/
```

## 说明

当前演示层默认使用轻量启发式模型，方便在没有深度学习环境的机器上立刻展示完整业务闭环。`models/` 和两个训练脚本保留 PyTorch BiLSTM 实现，用于训练真实模型并替换演示预测器。

推理层位于 `src/toxicity_predictor.py`，规则 fallback 位于 `src/toxicity_rules.py`：

```text
models/checkpoints/toxicity_bilstm.pt 存在
    -> 加载 BiLSTM checkpoint 推理

checkpoint 不存在或 PyTorch 不可用
    -> 使用 Demo Rule-based Fallback
```

页面侧边栏会显示当前实际使用的模型，避免演示叙述和代码实现不一致。

情感分类采用同样的运行时切换方式：

```text
models/checkpoints/sentiment_bilstm.pt 存在
    -> 加载 Sentiment BiLSTM checkpoint 推理

checkpoint 不存在或 PyTorch 不可用
    -> 使用 Demo Rule-based Fallback
```

训练脚本会输出 Accuracy、Precision、Recall、F1，并根据验证集 F1 保存最佳 checkpoint。checkpoint 同时保存模型配置，推理端不会再假设固定 hidden size 或 max length。

## 用户画像说明

`data/users.csv` 中的 `initial_topics` 只表示用户注册时选择的冷启动兴趣。只要用户存在历史行为，系统会根据：

```text
view / like / share / comment
    -> article.category / article.tags
    -> topic_preferences
```

动态计算兴趣。A1 / B / A2 只根据包含明确立场表达的评论生成，浏览、点赞、转发只影响话题兴趣，不直接提高立场置信度。

文章的 `category` 和 `tags` 用于兴趣画像，独立的 `main_topic` 用于观点倾向。一条评论只会归入对应文章的一个主话题，不会把同一立场复制到所有标签。用户级立场置信度综合历史样本量、观点一致性和单条证据强度；样本较少时，一致性贡献也会随样本量降低。

当前话题级立场的实现是“按文章 `main_topic` 对评论分组，再分别聚合用户的观点倾向”。`topic` 只提供少量文本相关性加成，这不是一个以“评论 + 话题”为联合输入训练的 stance detection 模型。规则层处理常见立场词、否定短语和词语重叠；更复杂的转折、反讽及多重否定应交给监督学习模型，而不是持续堆叠关键词。

Demo 中立场阈值、置信度、评论排序和推荐权重均为基于业务含义设置的原型参数，不代表训练得到的最优值。产品化时应使用人工标注验证集，把这些权重和阈值作为超参数，通过网格搜索、F1 和敏感性分析进行校准。

## GitHub 上传前清理

不要上传本地虚拟环境和缓存目录：

```bash
rm -rf .venv
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
```
