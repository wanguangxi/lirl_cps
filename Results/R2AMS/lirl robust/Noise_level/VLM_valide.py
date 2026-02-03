# ============================================================= #
#   QianWen-7B  +  LoRA  +  SGCL (Gumbel / GPU-SAT)           #
# ============================================================= #
# 导入必要的库
import torch, math, random, bitsandbytes as bnb  # bnb用于8位优化器，节省内存
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face的模型加载工具
from peft import LoraConfig, get_peft_model  # PEFT库用于参数高效微调（LoRA）
from quicksat_cuda import QuickSAT         # 自定义的GPU加速SAT求解器（通过pybind11封装）
from openbabel_cnf import smi2clauses_inc  # C++/CUDA实现的CNF（合取范式）转换器

# 设置使用GPU
device = "cuda"

# ---------------- 1. 加载基础模型并冻结参数 ------------------ #
# 使用阿里云的千问7B模型作为基础模型
model_name = "Qwen/QianWen-7B"
# 加载预训练模型，使用float16精度以节省显存，自动分配到可用设备
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto")
# 加载对应的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# 设置padding token为结束符token（用于批处理时对齐序列长度）
tokenizer.pad_token = tokenizer.eos_token  

# 冻结基础模型的所有参数（只训练LoRA适配器）
for p in model.parameters():          
    p.requires_grad_(False)

# ---------------- 2. 注入LoRA适配器 -------------------- #
# 配置LoRA参数
lora_cfg = LoraConfig(
    r=16,                    # LoRA的秩（rank），控制参数量
    lora_alpha=32,           # LoRA的缩放因子
    lora_dropout=0.05,       # dropout率，防止过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])  # 在注意力层的这些模块上添加LoRA
# 将LoRA配置应用到模型上，并设置为训练模式
model = get_peft_model(model, lora_cfg).train()     
# 打印可训练参数的统计信息
model.print_trainable_parameters()

# 创建优化器（仅优化LoRA参数）
# 使用8位AdamW优化器以节省内存
opt = bnb.optim.AdamW8bit(
    filter(lambda p: p.requires_grad, model.parameters()),  # 只选择需要梯度的参数
    lr=3e-5)  # 学习率

# ---------------- 3. Gumbel-Softmax采样器 ------------------ #
def gumbel_softmax_sample(logits, tau=0.5):
    """
    使用Gumbel-Softmax技巧进行可微分的离散采样
    Args:
        logits: 模型输出的logits (B, vocab_size)
        tau: 温度参数，控制采样的随机性（越小越接近argmax）
    Returns:
        y: 软化的one-hot向量 (B, vocab_size)
        tok: 采样得到的token id (B,)
    """
    # 生成Gumbel噪声：-log(-log(U))，其中U~Uniform(0,1)
    g = -torch.empty_like(logits).exponential_().log()  
    # 应用Gumbel-Softmax：将噪声加到logits上，再通过softmax
    y = torch.nn.functional.softmax((logits + g) / tau, dim=-1)
    # 获取最大概率对应的token（硬采样）
    tok = y.argmax(dim=-1)
    return y, tok

# ---------------- 4. QuickSAT设置 ------------------------- #
# 创建GPU加速的SAT求解器实例
# 每个GPU上下文拥有一个求解器，可以在批次间重复使用
solver = QuickSAT()  ####需要我们构建一个语法的验证求解器，这也只能验证PDDl文本的语法正不正确



# 增量式接口：add_clauses(list[list[int]])返回True（可满足）或False（不可满足）

# ---------------- 5. 训练循环（简化版） -------------- #
# 设置损失函数的权重
logic_weight = 4.0    # 逻辑约束损失的权重
kl_weight = 0.02      # KL散度损失的权重（防止偏离原模型太远）
clip_eps = 0.2        # 梯度裁剪阈值（未使用）
gumbel_tau = 0.5      # Gumbel-Softmax的温度参数

# 主训练循环，遍历自然语言提示的批次
for step, batch in enumerate(train_loader):          
    # 将文本提示转换为token ids
    ctx = tokenizer(batch["prompt"], return_tensors="pt",
                    padding=True).to(device)
    # 获取上下文的隐藏状态（不计算梯度）
    with torch.no_grad():
        ctx_out = model.model.forward(**ctx, use_cache=True).last_hidden_state
    
    # -------------------------------------------------- 解码循环
    B, T_ctx = ctx["input_ids"].shape  # B: 批次大小, T_ctx: 上下文长度
    past_kv = None                      # 缓存的键值对（用于加速自回归生成）
    logits_buf, token_buf = [], []      # 存储logits和tokens用于后续计算KL损失
    sat_flags = [1]*B                   # SAT状态标志：1表示满足，0表示违反
    core_sets = [ [] for _ in range(B) ]  # 存储每个样本的不满足核心（违反约束的token）
    solver.reset_batch(B)               # 重置求解器，为每个样本创建独立的求解器实例

    # 自回归生成max_gen_len个token
    for step_t in range(max_gen_len):
        # 使用Gumbel采样下一个token
        if step_t == 0:      # 第一步：使用上下文的最后一个隐藏状态
            hidden = ctx_out[:, -1]
        else:                # 后续步骤：使用上一步的输出
            hidden = model.model.layers[-1].output
        
        # 通过语言模型头获取词汇表上的logits
        lm_logits = model.lm_head(hidden)           # (B, |V|)
        # 使用Gumbel-Softmax进行采样
        y_soft, t_next = gumbel_softmax_sample(lm_logits, tau=gumbel_tau)
        # 保存生成的token和logits
        token_buf.append(t_next)
        logits_buf.append(lm_logits)

        # --- 对每个样本进行增量式SAT检查 --------
        for i in range(B):
            if sat_flags[i] == 0: continue  # 如果已经违反约束，跳过
            # 将新生成的token转换为CNF子句（逻辑约束） 
            #  
            #！！！！！如何将语法约束装换成逻辑约束？？？PDDL的逻辑约束，只能完整生成后再验证
            new_clauses = smi2clauses_inc(int(t_next[i].item()), i)
            # 添加新子句并检查可满足性
            sat = solver.add_clauses(i, new_clauses)
            if not sat:           # 如果违反约束
                sat_flags[i] = 0  # 标记为不满足
                core_sets[i] = solver.get_core(i)   # 获取不满足核心（导致冲突的token列表）
      
        # 将采样的token作为下一步的输入
        input_ids = t_next.unsqueeze(-1)
        # 继续前向传播，使用缓存的键值对加速计算
        outputs = model(input_ids=input_ids, past_key_values=past_kv,
                        use_cache=True)
        past_kv = outputs.past_key_values

    # -------- 6. 计算每个样本的损失 ----------------- #
    # 将所有步骤的logits堆叠并计算log概率
    logp = torch.stack(logits_buf, dim=1).log_softmax(-1)  # (B, L, V)
    # 获取实际采样的token的log概率
    logp_taken = torch.gather(logp, 2,
                  torch.stack(token_buf, dim=1).unsqueeze(-1)).squeeze(-1)

    # 任务损失（这里简化为负对数似然）
    L_task = -logp_taken.mean()

    # KL散度损失：防止微调后的模型偏离原始模型太远
    with torch.no_grad():
        # 使用冻结的基础模型计算参考logits
        ref_logits = model.get_base_model()(
            input_ids=torch.cat(token_buf,1),
            output_hidden_states=False).logits
    # 计算KL散度
    KL = torch.nn.functional.kl_div(
            logp, ref_logits.softmax(-1), log_target=True).mean()

    # 逻辑约束损失
    L_logic = torch.zeros((), device=device)
    for i in range(B):
        if sat_flags[i] == 1: continue  # 如果满足约束，跳过
        # 对不满足核心中的每个token，计算惩罚项
        #####核心#######
        for tok_id in core_sets[i]:
            # 获取该token的概率
            prob_i = torch.exp(logp_taken[i, tok_id])
            # 惩罚项：概率的倒数（概率越小，惩罚越大）
            L_logic = L_logic + (1.0 / prob_i)
    # 归一化并应用权重
    L_logic = logic_weight * L_logic / B

    # 总损失 = 任务损失 + KL正则化 + 逻辑约束损失
    loss = L_task + kl_weight*KL + L_logic

    # -------- 7. 优化器步骤 ------------------------------ #
    opt.zero_grad()      # 清零梯度
    loss.backward()      # 反向传播
    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()           # 更新参数

    # 每100步打印训练信息
    if step % 100 == 0:
        print(f"step {step}, loss={loss.item():.4f}, "
              f"logic viol.={(1-torch.tensor(sat_flags).float()).mean():.3f}")
