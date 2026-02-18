import math
import time
import numpy as np  
import torch
import transformers
from torch import nn

from datas import get_examples
from contextlib import contextmanager, nullcontext


# -----------------------------------------------------------------------------
# 1. 定义探针工具 (放在 blk_score 之前)
# -----------------------------------------------------------------------------
def _rel_l2(a, b, eps=1e-8):
    """
    计算相对 L2 误差。
    返回 Tensor (GPU) 以避免打断流水线。
    """
    # 1. 强制转 float32 防止半精度溢出
    a_f = a.float()
    b_f = b.float()
    # 2. 差异
    diff = a_f - b_f
    # 3. 使用高度优化的 linalg.vector_norm
    # dim=None 表示打平计算整个 Tensor 的范数
    num = torch.linalg.vector_norm(diff, ord=2)
    den = torch.linalg.vector_norm(a_f, ord=2).clamp_min(eps)
    return num / den


def _forward_one_layer(layer, x, attention_mask=None, position_ids=None, is_opt=False):
    if is_opt:
        return layer(x, attention_mask=attention_mask)[0]
    else:
        return layer(x, attention_mask=attention_mask, position_ids=position_ids)[0]

def _forward_range(layers, start, end, x, attention_mask=None, position_ids=None, is_opt=False):
    # run layers[start..end] sequentially, return list of outputs after each layer
    outs = []
    h = x
    for idx in range(start, end + 1):
        h = _forward_one_layer(layers[idx], h, attention_mask, position_ids, is_opt=is_opt)
        outs.append(h)
    return outs  # length = end-start+1


@contextmanager
def _perturb_layer_wanda(layer: nn.Module, wrapped_layers: dict, probe_ratio: float = 0.01):
    """
    Wanda 探针：临时屏蔽每一层中 Wanda Metric 最小的 probe_ratio 比例的权重
    """
    backups = []
    with torch.no_grad():
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                if name not in wrapped_layers:
                    continue

                # 获取 Wanda 需要的输入范数 scaler
                # 假设 BlockWanda 类中存储了 scaler_row
                if hasattr(wrapped_layers[name], 'scaler_row'):
                    scaler = wrapped_layers[name].scaler_row
                else:
                    continue  # 如果拿不到统计数据，跳过

                W = m.weight
                backups.append((W, W.data.clone()))

                if scaler.device != W.device:
                    scaler = scaler.to(W.device)

                # 计算 Wanda Metric: |W| * sqrt(scaler)
                wanda_metric = W.data.abs() * torch.sqrt(scaler.reshape(1, -1))
                flat_metric = wanda_metric.view(-1)

                k = int(probe_ratio * flat_metric.numel())
                if k > 0:
                    thresh = torch.kthvalue(flat_metric, k).values
                    mask = (wanda_metric > thresh).to(W.data.dtype)
                    W.data.mul_(mask)
    try:
        yield
    finally:
        with torch.no_grad():
            for W, orig in backups:
                W.data.copy_(orig)


@contextmanager
def _perturb_layer_magnitude(layer: nn.Module, probe_ratio: float = 0.01):
    """
    Magnitude 探针：临时屏蔽绝对值最小的权重
    """
    backups = []
    with torch.no_grad():
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                W = m.weight
                backups.append((W, W.data.clone()))

                flat = W.data.abs().view(-1)
                k = int(probe_ratio * flat.numel())
                if k > 0:
                    thresh = torch.kthvalue(flat, k).values
                    mask = (W.data.abs() > thresh).to(W.data.dtype)
                    W.data.mul_(mask)
    try:
        yield
    finally:
        with torch.no_grad():
            for W, orig in backups:
                W.data.copy_(orig)


def get_perturb_context(layer, method, wrapped_layers=None, probe_ratio=0.02):
    """
    根据 method 返回对应的扰动上下文管理器。
    支持: 'wanda', 'magnitude', 'none'
    """
    if method == "wanda":
        if wrapped_layers is None:
            raise ValueError("Wanda pruning requires 'wrapped_layers' to be passed.")
        return _perturb_layer_wanda(layer, wrapped_layers, probe_ratio=probe_ratio)

    elif method == "magnitude":
        # Magnitude 不需要 wrapped_layers
        return _perturb_layer_magnitude(layer, probe_ratio=probe_ratio)

    elif method == "none":
        # 不做任何扰动，用于测试基准
        return nullcontext()

    else:
        raise ValueError(f"Unknown pruning method: {method}")



def combine_metrics(D, Rn, alpha, mode="prod"):
    """
    将局部指标 D 和全局风险指标 Rn 结合的封装函数

    Args:
        D (np.ndarray): 原始离群值指标 (Difficulty)
        Rn (np.ndarray): 归一化后的风险指标 (Normalized Risk, 0~1)
        alpha (float): 结合系数 (Weight 或 Gain)
        mode (str): 结合模式 ('linear', 'prod', 'geom', 'gate', 'max', 'exp')

    Returns:
        np.ndarray: 结合后的最终指标 D_hat
    """
    if mode == "linear":
        # 线性加权: (1 - α) * D + α * Rn
        return (1 - alpha) * D + alpha * Rn

    elif mode == "prod":
        # 增益乘法: D * (1 + α * Rn)
        return D * (1 + alpha * Rn)

    elif mode == "geom":
        # 加权几何平均: D^(1-α) * (Rn)^α
        # 使用 epsilon 防止 Rn 为 0 时结果坍缩
        return np.power(D, 1 - alpha) * np.power(Rn + 1e-6, alpha)

    elif mode == "gate":
        # 软阈值保护 (Sigmoid Gating)
        # k 控制陡峭度, tau 控制开启阈值
        k, tau = 10, 0.5
        gate = 1 / (1 + np.exp(-k * (Rn - tau)))
        return D * (1 + alpha * gate)

    elif mode == "max":
        # 竞争模式: 取两者较大值
        return np.maximum(D, alpha * Rn)

    elif mode == "exp":
        # 指数增强: D * exp(α * Rn)
        return D * np.exp(alpha * Rn)

    else:
        # 默认回退到 prod 模式或抛出错误
        print(f"[WARNING] Unknown combine_mode '{mode}', falling back to 'prod'.")
        return D * (1 + alpha * Rn)


# -----------------------------------------------------------------------------
# 1. end
# -----------------------------------------------------------------------------



def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_input(model, layers, dataloader, device):
    model.seqlen = 2048
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                cache['position_ids'] = kwargs['position_ids']
            if "position_embeddings" in kwargs and kwargs['position_embeddings'] is not None:
                cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.unsqueeze(0).to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    del cache["i"]

    return inps, outs, cache


class BlockWanda:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def blk_s(
            self, block_size=128, s=0.5
    ):
        weight = self.layer.weight.to(dtype=torch.float32)
        co, ci = weight.shape
        sx = self.H.to(dtype=torch.float32)
        # sw = weight.T @ weight
        #
        # s = sx * sw
        # print(s[1:17, 1:17])
        # input()

        if block_size == 0:
            block_size = ci

        score = (weight ** 2) * torch.diag(sx)
        # scale = torch.max(score)

        re_construct = 0.
        # s = 0.5
        prune_num = int(block_size * s)
        prune_idx = []
        for i1 in range(0, ci, block_size):
            i2 = min(i1 + block_size, ci)

            w1 = weight[:, i1:i2]
            w2 = weight[:, i2:]
            score1 = score[:, i1:i2]
            score2 = score[:, i2:]
            sx1 = sx[i1:i2, i1:i2]
            sx2 = sx[i1:i2, i2:]

            err = torch.zeros_like(w1)
            for i in range(prune_num):
                idx = torch.argmin(score1, dim=1).unsqueeze(1)
                re_construct += torch.sum(score1.gather(1, idx))
                prune_idx.append(idx + i1)

                w = w1.gather(1, idx)
                change = w1 * w * sx1[idx.squeeze(1)]
                score1 += 2 * change
                score1.scatter_(dim=1, index=idx, value=torch.inf)
                err.scatter_(dim=1, index=idx, src=w)

            change = w2 * (err @ sx2)
            score2 += 2 * change

        # re_construct /= (co * ci * 0.5)
        return re_construct

    def wanda_s(self, block_size=128):
        weight = self.layer.weight
        sx = self.H

        metric = weight.abs() * (torch.diag(sx) ** 0.5)
        return torch.mean(metric)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def blk_score(args, model, tokenizer, alpha, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=2048)
    print("dataset loading complete")

    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
    elif "Qwen" in args.base_model:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    start_time = time.time()
    with torch.no_grad():
        inps, outs, cache = prepare_calibration_input(model, layers, dataloader, device)

    all_layer_ratio = []
    all_layer_metric = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            for key, value in cache.items():
                if isinstance(value, tuple):
                    cache[key] = tuple([v.to(dev) for v in value])
                else:
                    cache[key] = value.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BlockWanda(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        print(f"get layer sp layer {i}")
        for name in subset:
            W_metric = wrapped_layers[name].blk_s(block_size=args.block, s=args.resp)
            wrapped_layers[name].free()

            layer_wmetric.append(W_metric)

        for j in range(args.num_examples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        all_layer_metric.append(layer_wmetric)
        all_layer_ratio.append(torch.mean(layer_wmetric.float()).abs())

        torch.cuda.empty_cache()

    all_layer_ratio = torch.tensor(all_layer_ratio)
    # print(torch.stack(all_layer_metric, dim=0))
    print(all_layer_ratio / torch.sum(all_layer_ratio))
    layer_imp = 1 - all_layer_ratio / torch.sum(all_layer_ratio)  # layer imp
    layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min()) * alpha * 2
    # layer_imp = layer_imp / torch.sum(layer_imp)
    # layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min())
    all_layer_ratio = args.final_s + torch.mean(layer_imp) - layer_imp

    # softmax 减小方差
    # while any((all_layer_ratio < 0) | (all_layer_ratio > 1)):
    #     layer_imp = torch.softmax(layer_imp, dim=0)
    #     all_layer_ratio = args.final_s + torch.mean(layer_imp) - layer_imp

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    print(all_layer_ratio)
    return all_layer_ratio.tolist()


def blk_score_with_risk(args, model, tokenizer, alpha, device=torch.device("cuda:0")):
    # --- [1. 参数初始化] ---
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # Risk 相关参数 (建议在 args 中定义，这里提供默认值)
    risk_k         = args.risk_k
    risk_alpha     = args.risk_alpha
    risk_decay     = args.risk_decay
    risk_probe     = args.risk_probe
    risk_nsamples  = args.risk_nsamples
    combine_mode   = args.combine_mode
    include_self   = args.risk_include_self
    Lambda = args.Lambda
    # print(f"Risk Config: k={risk_k}, alpha={risk_alpha}, mode={combine_mode}, probe={risk_probe}")

    # --- [2. 数据准备] ---
    print("loading calibration data")
    dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=2048)
    
    # 识别模型类型并获取 Layers
    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
        is_opt = False
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
        is_opt = True
    elif "Qwen" in args.base_model:
        layers = model.transformer.h
        is_opt = False
    else:
        layers = model.model.layers
        is_opt = False

    start_time = time.time()
    # 预计算第一层的输入
    with torch.no_grad():
        inps, outs, cache = prepare_calibration_input(model, layers, dataloader, device)

    # 存储原始指标
    D_raw_list = [] # 本地 LSA 指标 (Difficulty)
    R_raw_list = [] # 全局风险指标 (Risk)

    # --- [3. 主循环：逐层扫描] ---
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # 处理多卡/设备映射
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            for key, value in cache.items():
                if isinstance(value, tuple):
                    cache[key] = tuple([v.to(dev) for v in value])
                else:
                    cache[key] = value.to(dev)
        
        # 3.1 包装层并挂载 Hook
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BlockWanda(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # 3.2 第一次 Forward (Pass 1): 收集 Hessian (self.H)
        # 这一步是为了填满 BlockWanda 中的 self.H，供 blk_s 和 Risk Probe 使用
        for j in range(args.num_examples):
            with torch.no_grad():
                if is_opt:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=cache.get('attention_mask'))[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        
        for h in handles: h.remove()

        print(f"Analyzing layer {i} / {len(layers)} ...")

        # ------------------------------------------------------
        # [A] 准备 Probe 数据
        # ------------------------------------------------------
        # Wanda Probe 需要 scaler_row (diag(X^TX))。
        # BlockWanda 已经计算了 H (X^TX)，我们直接提取对角线即可。
        for name in subset:
            wrapped_layers[name].scaler_row = torch.diag(wrapped_layers[name].H)

        # ------------------------------------------------------
        # [B] 计算 Risk Metric (Global Propagation)
        # ------------------------------------------------------
        risk_score = 0.0
        # 只有当 alpha > 0 且不是最后一层时，才有计算传播风险的意义
        # (最后一层也可以算，看是否 include_self)
        if risk_alpha > 0:
            current_risk_samples = min(risk_nsamples, args.num_examples)
            # 确定观察窗口: [i, i + k]
            j_end = min(len(layers) - 1, i + risk_k)
            
            total_weighted_drift = 0.0
            total_weight = 0.0

            # 遍历样本
            for j in range(current_risk_samples):
                x0 = inps[j].unsqueeze(0) # 当前层输入

                # 1. 基准运行 (Base Run)
                base_outs = _forward_range(layers, i, j_end, x0, is_opt=is_opt, **cache)

                # 2. 扰动运行 (Perturbed Run)
                # 使用上下文管理器临时 Mask 掉当前层 1% 的权重
                ctx = get_perturb_context(
                    layer=layers[i], 
                    method="wanda", 
                    wrapped_layers=wrapped_layers, 
                    probe_ratio=risk_probe
                )
                
                with ctx:
                    pert_outs = _forward_range(layers, i, j_end, x0, is_opt=is_opt, **cache)
                
                # 3. 计算加权漂移 (Weighted Drift)
                # 确定起始比较层 t
                # base_outs[0] 是 Layer i 的输出
                # base_outs[1] 是 Layer i+1 的输出
                if include_self:
                    start_t = 0
                else:
                    # 如果只有一层(即最后一层)，强制设为0，否则循环不执行，Risk为0
                    start_t = 0 if len(base_outs) == 1 else 1

                for t in range(start_t, len(base_outs)):
                    # t 表示距离当前层的深度
                    w = risk_decay ** t
                    
                    # 使用你提供的 _rel_l2 计算误差
                    drift = _rel_l2(base_outs[t], pert_outs[t])
                    
                    total_weighted_drift += w * drift
                    total_weight += w
            
            # 计算平均 Risk
            if total_weight > 0:
                # 注意：这里 drift 已经是 tensor 了，取 item 转 float
                risk_score = (total_weighted_drift / total_weight).item()
        
        R_raw_list.append(risk_score)

        # ------------------------------------------------------
        # [C] 计算 Local Metric (D) - 使用原 blk_s
        # ------------------------------------------------------
        layer_local_metrics = []
        for name in subset:
            # blk_s 计算的是重建误差 (Reconstruction Error)
            W_metric = wrapped_layers[name].blk_s(block_size=args.block, s=args.resp)
            
            # 必须在 Risk 计算完之后再 free，因为 Risk 上下文需要 scaler_row
            wrapped_layers[name].free() 
            layer_local_metrics.append(W_metric)

        # 聚合当前层所有模块的误差 (mean)
        # 将 list of tensors 展平 -> cat -> mean -> abs
        flat_metrics = torch.cat([torch.flatten(x.cpu()) for x in layer_local_metrics])
        local_score_val = torch.mean(flat_metrics.float()).abs().item()
        D_raw_list.append(local_score_val)

        # 3.3 第二次 Forward: 为下一层准备输入
        # 因为 hooks 被移除了，所以得重新跑一遍纯净的 forward 来拿 outputs
        for j in range(args.num_examples):
             with torch.no_grad():
                if is_opt:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=cache.get('attention_mask'))[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        
        # Swap inputs
        inps, outs = outs, inps
        torch.cuda.empty_cache()

    # --- [4. 指标融合与比例分配] ---
    
    # 转换为 Numpy 数组以便使用 combine_metrics
    D = np.array(D_raw_list, dtype=np.float32) # Shape: [num_layers]
    R = np.array(R_raw_list, dtype=np.float32) # Shape: [num_layers]

    print("\n--- Metrics Summary ---")
    print(f"D (Local) range: {D.min():.4e} ~ {D.max():.4e}")
    print(f"R (Risk)  range: {R.min():.4e} ~ {R.max():.4e}")

    # --- [4. 归一化与方向对齐] ---

    # 4.1 归一化 D (保持方向: D 越大 -> 这一层越"重" -> 我们希望剪越多)
    if D.max() - D.min() > 1e-9:
        D_norm = (D - D.min()) / (D.max() - D.min())
    else:
        D_norm = np.zeros_like(D)

    # 4.2 归一化 R 并反转 (Risk 越大 -> 越危险 -> 剪越少)
    # 我们定义 R_safe (安全性): Risk 越低 -> Safety 越高 -> 我们希望剪越多
    if R.max() - R.min() > 1e-9:
        R_norm = (R - R.min()) / (R.max() - R.min())
    else:
        R_norm = np.zeros_like(R)
    
    # 【关键修改】反转方向：R_safe 越高代表越安全，方向与 D 一致了
    R_safe = 1.0 - R_norm

    # --- [5. 线性结合] ---

    # 线性结合：总分 = LSA分数(想多剪) + alpha * 安全分数(想多剪)
    # Combined Score 越高，代表这一层 "既是大层，又很安全"，最适合被剪掉
    combined_score_numpy = combine_metrics(
        D=D_norm, 
        Rn=R_safe, 
        alpha=args.risk_alpha, 
        mode=args.combine_mode 
    )

    # 转回 Tensor
    all_layer_metric_final = torch.from_numpy(combined_score_numpy).to(device)

    # --- [6. 最终计算 Sparsity Ratio] ---

    # 1. 再次归一化 Combined Score 到 [0, 1]
    metrics_norm = (all_layer_metric_final - all_layer_metric_final.min()) / \
                   (all_layer_metric_final.max() - all_layer_metric_final.min() + 1e-8)

    # 2. 计算 Imp (Importance)
    # 回顾分配公式: Ratio = Base + (Mean_Imp - Imp)
    # 我们希望: Score 高 (Metrics_Norm 大) -> Ratio 高 (剪得多)
    # 要让 Ratio 变大，(Mean - Imp) 必须是正数 -> Imp 必须小
    # 所以: Imp 与 Score 成反比
    layer_imp = 1.0 - metrics_norm 

    # 3. 调整强度 (Lambda 参数控制方差)
    # 这里的 Lambda 是函数入参 (控制层间差异幅度)
    layer_imp = (layer_imp - layer_imp.min()) / \
                (layer_imp.max() - layer_imp.min() + 1e-8) * Lambda * 2 

    # 4. 分配稀疏度
    # 验证逻辑: 
    #   High D / High Safety -> High Score -> Low Imp -> (Mean - Low) > 0 -> Ratio 增加 (剪得多) ✅
    #   Low D / High Risk    -> Low Score  -> High Imp -> (Mean - High) < 0 -> Ratio 减少 (剪得少) ✅
    all_layer_ratio = args.final_s + torch.mean(layer_imp) - layer_imp

    # 5. 截断保护
    all_layer_ratio = torch.clamp(all_layer_ratio, 0.0, 1.0)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    
    return all_layer_ratio.tolist()



### block?? multiply??
def blk_score_global(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=2048)
    print("dataset loading complete")

    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    start_time = time.time()
    with torch.no_grad():
        inps, outs, cache = prepare_calibration_input(model, layers, dataloader, device)

    all_layer_ratio = []
    all_layer_metric = []
    all_layer_numel = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_numel = [subset[name].weight.numel() for name in subset]
        all_layer_numel.append(torch.tensor(layer_numel))

        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            for key, value in cache.items():
                if isinstance(value, tuple):
                    cache[key] = tuple([v.to(dev) for v in value])
                else:
                    cache[key] = value.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BlockWanda(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        print(f"get layer sp layer {i}")
        for name in subset:
            W_metric = wrapped_layers[name].blk_s(block_size=args.block)
            wrapped_layers[name].free()

            layer_wmetric.append(W_metric)

        for j in range(args.num_examples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        all_layer_metric.append(layer_wmetric)
        all_layer_ratio.append(torch.mean(layer_wmetric.float()).abs())

        torch.cuda.empty_cache()

    ## mean
    all_layer_metric = torch.stack(all_layer_metric, dim=0)
    # print(all_layer_metric)
    if args.layer == "blkb":
        attn_metric = torch.mean(all_layer_metric[:, :4], dim=1, keepdim=True)
        ffn_metric = torch.mean(all_layer_metric[:, 4:], dim=1, keepdim=True)
        all_layer_metric[:, :4] = attn_metric @ torch.ones(1, 4)
        all_layer_metric[:, 4:] = ffn_metric @ torch.ones(1, all_layer_metric.shape[1] - 4)
        # print(all_layer_metric)

    all_layer_numel = torch.stack(all_layer_numel, dim=0)
    layer_imp = 1 - all_layer_metric / torch.sum(all_layer_metric)  # layer imp
    layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min()) * args.Lamda * 2
    # print(layer_imp)
    layer_prune_numel = all_layer_numel * args.final_s + (torch.mean(layer_imp) - layer_imp) * torch.mean(
        all_layer_numel.float())
    all_layer_ratio = layer_prune_numel / all_layer_numel
    # all_layer_ratio = args.final_s + torch.mean(layer_imp) - layer_imp

    # component wise llama2-7 37.28
    # all_layer_metric = torch.stack(all_layer_metric, dim=0)
    # all_layer_numel = torch.stack(all_layer_numel, dim=0)
    # layer_imp = 1 - all_layer_metric / torch.sum(all_layer_metric)
    # layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min()) * args.Lamda * 2
    #
    # layer_prune_numel = torch.sum(all_layer_numel * args.final_s)
    # bias_numel = torch.sum(all_layer_numel * (torch.mean(layer_imp) - layer_imp))
    # p0 = (layer_prune_numel - bias_numel) / torch.sum(all_layer_numel)
    # all_layer_ratio = p0 + torch.mean(layer_imp) - layer_imp

    print(all_layer_ratio)
    print(torch.min(all_layer_ratio), torch.max(all_layer_ratio))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    return all_layer_ratio.tolist()


class BLK_v1:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = [2, 0.06, 0.02, 0.04, 0.02, 0.04, 0.1, 0.15, 0.12]  # zero index for n:m

    @torch.no_grad()
    def get_layer_sp(self, args):
        self.alpha[0] = args.Lamda
        if args.layer == 'lsa':
            return blk_score_with_risk(args, self.model, self.tokenizer, self.alpha[int(args.final_s * 10)],
                             device=self.model.device)
        elif args.layer == "lsab" or args.layer == "lsac":
            return blk_score_global(args, self.model, self.tokenizer,
                                    device=self.model.device)
        else:
            raise NotImplementedError
