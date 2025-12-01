# 最小化集成方案 - 将动作头加到现有VLM

如果你已经有了VLM，只需要3个核心步骤：

## 📦 需要的文件

**核心文件**（必需）:
1. `action_tokenizer_6d.py` - 动作令牌化器
2. `action_processor.py` - 动作归一化/反归一化

**可选文件**:
- `action_config.json` - 配置文件
- `action_statistics_template.json` - 统计信息模板

## 🔧 集成步骤

### 步骤1: 扩展词汇表并添加embedding层

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from action_tokenizer_6d import SpatialActionTokenizer6D

# 1. 加载你的VLM模型
your_vlm = YourVLMModel.from_pretrained("your-model-path")
tokenizer = AutoTokenizer.from_pretrained("your-model-path")

# 2. 创建action tokenizer
action_config = {
    "num_bins": {
        "translation": {"theta_bins": 16, "phi_bins": 32, "r_bins": 8},
        "rotation": {"roll_bins": 16, "pitch_bins": 16, "yaw_bins": 16}
    }
}

action_tokenizer = SpatialActionTokenizer6D(
    tokenizer=tokenizer,
    num_bins=action_config["num_bins"],
    use_spherical=True
)

print(f"新增动作令牌数: {action_tokenizer.vocab_size}")  # 8192
print(f"新的词汇表大小: {len(tokenizer)}")

# 3. 扩展模型的token embeddings
your_vlm.resize_token_embeddings(len(tokenizer))

# 4. 添加spatial embedding层（推荐但可选）
your_vlm.spatial_embed_tokens = nn.Embedding(
    action_tokenizer.vocab_size,  # 8192
    your_vlm.config.hidden_size   # 你的模型隐藏层维度
)

# 5. 记录动作令牌的起始索引
your_vlm.action_token_begin_idx = action_tokenizer.action_token_begin_idx
```

### 步骤2: 修改forward函数

在你的VLM的forward函数中添加对动作令牌的特殊处理：

```python
def forward(self, input_ids, pixel_values=None, labels=None, **kwargs):
    # 获取标准的token embeddings
    inputs_embeds = self.get_input_embeddings()(input_ids)

    # ⭐ 新增：如果有spatial_embed_tokens，替换动作令牌的embedding
    if hasattr(self, 'spatial_embed_tokens') and self.spatial_embed_tokens is not None:
        # 识别动作令牌的位置
        action_mask = (input_ids >= self.action_token_begin_idx) & \
                      (input_ids < self.action_token_begin_idx + self.spatial_embed_tokens.num_embeddings)

        if action_mask.any():
            # 计算相对索引
            relative_indices = input_ids[action_mask] - self.action_token_begin_idx
            # 替换为spatial embeddings
            inputs_embeds[action_mask] = self.spatial_embed_tokens(relative_indices)

    # 处理图像（如果有）
    if pixel_values is not None:
        image_embeds = self.vision_encoder(pixel_values)
        # ... 将图像特征融合到inputs_embeds中

    # 继续正常的forward流程
    outputs = self.language_model(inputs_embeds=inputs_embeds, **kwargs)

    # 计算损失
    if labels is not None:
        logits = outputs.logits
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}

    return outputs
```

### 步骤3: 训练和推理

#### 训练时：

```python
from action_processor import ActionProcessor
import numpy as np

# 创建action processor
action_processor = ActionProcessor(
    action_tokenizer=action_tokenizer,
    statistics=your_action_statistics  # 从数据计算得到
)

# 准备训练数据
for batch in dataloader:
    image = batch['image']
    text = batch['instruction']
    action = batch['action']  # (batch_size, 6)

    # 编码动作为令牌
    action_tokens = action_processor.encode_actions_for_training(
        action, dataset_key="train"
    )

    # 将动作令牌添加到输入序列
    full_text = text + "".join(action_tokens)

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt")

    # 创建labels（只计算动作token的损失）
    labels = inputs["input_ids"].clone()
    labels[:, :-len(action_tokens)] = -100  # 忽略非动作部分

    # Forward
    outputs = your_vlm(
        input_ids=inputs["input_ids"],
        pixel_values=image,
        labels=labels
    )

    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
```

#### 推理时：

```python
# 准备输入
text = "Pick up the red cube"
inputs = tokenizer(text, return_tensors="pt")

# 生成动作令牌（自回归生成2个token）
output_ids = your_vlm.generate(
    input_ids=inputs["input_ids"],
    pixel_values=image,
    max_new_tokens=2,  # 6D动作需要2个token
    do_sample=False
)

# 提取动作token
action_token_ids = output_ids[:, -2:]  # 最后2个token

# 解码为连续动作
action_token_ids_np = action_token_ids.cpu().numpy()
normalized_action = action_tokenizer.decode_token_ids_to_actions(
    action_token_ids_np
)  # (1, 6) in [-1, 1]

# 反归一化到原始尺度
action = action_processor.unnormalize_actions(
    normalized_action, dataset_key="train"
)

print(f"预测的动作: {action}")  # [x, y, z, roll, pitch, yaw]
```

## 🎯 关键点说明

### 1. Spatial Embedding层的作用

**有spatial_embed_tokens**:
```
动作令牌 → spatial_embed_tokens → 独立学习的embedding → 更好的动作表示
```

**没有spatial_embed_tokens**:
```
动作令牌 → 共享的token_embeddings → 与文本共享embedding空间
```

**推荐**: 使用spatial_embed_tokens，因为：
- 动作和语言是不同的模态
- 独立的embedding可以学习更好的动作表示
- 这是SpatialVLA的做法，经过验证有效

### 2. 是否需要从SpatialVLA加载权重？

**如果你的VLM已经很强大**（如Qwen-VL-7B）:
- ✅ 不加载SpatialVLA权重，随机初始化spatial_embed_tokens
- ✅ 只需要借用tokenizer和bin策略
- ✅ 在你的数据上从头训练动作头

**如果你想利用SpatialVLA的动作知识**:
- 可以加载spatial_embed_tokens的权重
- 但需要注意：SpatialVLA是8194个token（含gripper），你是8192个
- 只加载前8192个token的权重即可

### 3. 最小改动方案

如果你不想修改VLM的forward，可以：

```python
# 训练时：直接在输入中使用动作令牌
# 推理时：把生成的token ID传给action_tokenizer解码

# 不需要spatial_embed_tokens！
# 动作令牌就像普通token一样处理

# 优点：改动最小
# 缺点：动作表示可能不够好
```

## 📝 完整最小示例

```python
# ===== 文件: minimal_integration.py =====

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from action_tokenizer_6d import SpatialActionTokenizer6D
from action_processor import ActionProcessor, compute_action_statistics

# 1. 准备
your_vlm = load_your_vlm()  # 你的VLM
tokenizer = load_your_tokenizer()

# 2. 创建action tokenizer
action_tokenizer = SpatialActionTokenizer6D(
    tokenizer=tokenizer,
    num_bins={
        "translation": {"theta_bins": 16, "phi_bins": 32, "r_bins": 8},
        "rotation": {"roll_bins": 16, "pitch_bins": 16, "yaw_bins": 16}
    },
    use_spherical=True
)

# 3. 扩展模型
your_vlm.resize_token_embeddings(len(tokenizer))
your_vlm.spatial_embed_tokens = nn.Embedding(
    action_tokenizer.vocab_size,
    your_vlm.config.hidden_size
)
your_vlm.action_token_begin_idx = action_tokenizer.action_token_begin_idx

# 4. 计算数据统计（一次性）
all_actions = np.array([...])  # 你的所有训练动作
statistics = compute_action_statistics(all_actions, "my_dataset")

# 5. 创建processor
action_processor = ActionProcessor(action_tokenizer, statistics)

# 6. 训练
for batch in dataloader:
    # 编码动作
    action_tokens = action_processor.encode_actions_for_training(
        batch['action'], "my_dataset"
    )

    # 构造输入
    text_with_action = batch['text'] + "".join(action_tokens)
    inputs = tokenizer(text_with_action, return_tensors="pt")

    # 在forward中处理spatial_embed_tokens（见步骤2）
    outputs = your_vlm(inputs["input_ids"], batch['image'])

    loss = outputs.loss
    loss.backward()
    optimizer.step()

# 7. 推理
output_ids = your_vlm.generate(
    tokenizer(text, return_tensors="pt")["input_ids"],
    image,
    max_new_tokens=2
)

action_tokens = output_ids[:, -2:].cpu().numpy()
action = action_tokenizer.decode_token_ids_to_actions(action_tokens)
action = action_processor.unnormalize_actions(action, "my_dataset")

print(f"预测动作: {action}")
```

## 🚀 总结

**你确实只需要**:
1. ✅ `action_tokenizer_6d.py` - 核心令牌化逻辑
2. ✅ `action_processor.py` - 数据预处理
3. ✅ 在VLM中添加spatial_embed_tokens层（可选但推荐）
4. ✅ 修改forward处理动作令牌（如果用spatial_embed）

**不需要**:
- ❌ `qwen_vl_with_action.py` - 如果你不用Qwen-VL
- ❌ `transfer_weights.py` - 如果不从SpatialVLA迁移
- ❌ 示例训练/推理代码 - 如果你有自己的训练流程

核心就是：**扩展词汇表 → 添加embedding层 → 正常训练/推理**
