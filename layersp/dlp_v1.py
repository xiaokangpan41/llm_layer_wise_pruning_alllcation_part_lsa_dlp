import time

import torch
from torch import nn
import numpy as np

from datas import get_examples
from contextlib import contextmanager, nullcontext

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


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

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

@contextmanager
def _perturb_layer_magnitude(layer: nn.Module, probe_ratio: float = 0.02):
    """
    Temporarily zero out the smallest |w| weights in every nn.Linear in this layer.
    This is ONLY for risk probing, not final pruning. 模拟“Magnitude Pruning 对后续的影响”。
    """
    backups = []
    with torch.no_grad():
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                W = m.weight
                backups.append((W, W.data.clone()))

                flat = W.data.abs().view(-1)
                k = int(probe_ratio * flat.numel())
                if k <= 0:
                    continue
                thresh = torch.kthvalue(flat, k).values
                mask = (W.data.abs() > thresh).to(W.data.dtype)
                W.data.mul_(mask)
    try:
        yield
    finally:
        with torch.no_grad():
            for W, orig in backups:
                W.data.copy_(orig)

@contextmanager
def _perturb_layer_wanda(layer: nn.Module, wrapped_layers: dict, probe_ratio: float = 0.02):
    """
    Risk Probing (Wanda Variant):
    临时屏蔽每一层中 Wanda Metric ( |W| * |X| ) 最小的权重。

    Args:
        layer: 当前层对象 (e.g., LlamaDecoderLayer)
        wrapped_layers: 包含 scaler_row 的字典, key 是层内模块名 (e.g., "self_attn.q_proj")
        probe_ratio: 扰动比例
    """
    backups = []

    # 确保不计算梯度
    with torch.no_grad():
        # 我们使用 named_modules 来匹配 wrapped_layers 中的名字
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                # 1. 检查该层是否有对应的输入统计信息 (scaler)
                if name not in wrapped_layers:
                    continue

                scaler = wrapped_layers[name].scaler_row

                # 2. 准备数据
                W = m.weight
                backups.append((W, W.data.clone())) # 备份

                # 3. 计算 Wanda Metric: |W| * sqrt(scaler)
                # scaler 形状通常是 [in_features], 需要 reshape 为 [1, in_features] 进行广播
                # 确保 scaler 和 W 在同一个 device
                if scaler.device != W.device:
                    scaler = scaler.to(W.device)

                # Metric 计算 (Wanda 核心)
                wanda_metric = W.data.abs() * torch.sqrt(scaler.reshape(1, -1))
                flat_metric = wanda_metric.view(-1)

                # 4. 确定阈值并生成 Mask
                k = int(probe_ratio * flat_metric.numel())
                if k > 0:
                    # 找出第 k 小的值作为阈值
                    thresh = torch.kthvalue(flat_metric, k).values

                    # 生成 Mask (保留 Metric 大于阈值的权重)
                    mask = (wanda_metric > thresh).to(W.data.dtype)

                    # 5. 应用临时 Mask
                    W.data.mul_(mask)

    try:
        yield
    finally:
        # 6. 恢复原始权重
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

def dlp_score(args, model, tokenizer, alpha, device=torch.device("cuda:0")):
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

    D_list = []
    R_raw = []

    # --- Hyperparameters ---
    risk_k = args.risk_k
    risk_probe = args.risk_probe
    risk_decay = args.risk_decay
    total_nsamples = args.nsamples
    risk_nsamples = args.risk_nsamples

    probe_method = args.probe_method
    attention_mask = cache.get("attention_mask", None)
    position_ids = cache.get("position_ids", None)
    is_opt = "opt" in args.base_model.lower()

    # PHASE 1: Compute the importance score for each weight in each layer
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
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name, wrapped_layers_ref):
            def tmp(_, inp, out):
                wrapped_layers_ref[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name, wrapped_layers)))
        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        print(f"get layer sp layer {i} with dlp")
        for name in subset:
            W = torch.abs(subset[name].weight.data).to(dtype=torch.float32)
            X = wrapped_layers[name].scaler_row.reshape((1, -1)).to(dtype=torch.float32)
            W_metric = W * torch.sqrt(X)

            layer_wmetric.append(W_metric)

        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        D_list.append(torch.median(layer_wmetric.float()).abs())
        
        torch.cuda.empty_cache()

        # PHASE 2: Compute the risk score for each layer based on the importance scores
        j_end = min(len(layers) - 1, i + risk_k)
        risk_sum = torch.tensor(0.0, device=inps.device)
        w_sum = 0.0

        current_risk_nsamples = min(risk_nsamples, total_nsamples, args.num_examples)

        for j in range(current_risk_nsamples):
            x0 = inps[j].unsqueeze(0)

            # ---------------------------------------------------------------------
            # A. Base Run (基准运行)
            # ---------------------------------------------------------------------
            # 统一使用 j_end，_forward_range 会返回从 i 到 j_end 的所有层输出
            # base_outs[0] 是当前层 i 的输出，base_outs[1] 是 i+1 层输出...
            base_outs = _forward_range(
                layers, i, j_end, x0,
                attention_mask=attention_mask, position_ids=position_ids, is_opt=is_opt
            )

            # ---------------------------------------------------------------------
            # B. Perturbed Run (扰动运行)
            # ---------------------------------------------------------------------
            ctx = get_perturb_context(
                layer=layers[i],
                method=probe_method,
                wrapped_layers=wrapped_layers,
                probe_ratio=risk_probe
            )

            with ctx:
                pert_outs = _forward_range(
                    layers, i, j_end, x0,
                    attention_mask=attention_mask, position_ids=position_ids, is_opt=is_opt
                )

            # ---------------------------------------------------------------------
            # C. Calculate Drift (计算漂移 - 健壮版)
            # ---------------------------------------------------------------------

            # 1. 确定起始索引 (start_t)
            # getattr 提供了默认值 True，防止 args 里没定义这个参数报错
            include_self = args.risk_include_self

            if include_self:
                start_t = 0
            else:
                # 如果不包含当前层，通常从 t=1 (下一层) 开始
                # 【关键兜底】：如果是最后一层 (len==1)，没有下一层可看，必须强制看自己 (t=0)
                # 否则 w_sum 为 0，Risk 为 0，导致最后一层被误剪
                if len(base_outs) == 1:
                    start_t = 0
                else:
                    start_t = 1

            # 2. 统一循环累加
            # len(base_outs) 自动适应了是中间层还是最后一层
            for t in range(start_t, len(base_outs)):
                # t 代表距离当前层的步数 (0=当前, 1=下一层...)

                # 计算权重: 距离越远，权重越小 (decay^0=1, decay^1=0.8...)
                w = (risk_decay ** t)

                # 计算两组输出的相对 L2 误差
                d = _rel_l2(base_outs[t], pert_outs[t])

                # 累加加权误差
                risk_sum += w * d
                w_sum += w

        if w_sum == 0:
            R_i = 0.0
        else:
            # risk_sum 是 tensor, w_sum 是 float
            R_i = (risk_sum / w_sum).item()

        R_raw.append(R_i)

        del wrapped_layers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # Combine D and R into a single importance score for each layer
    D = np.array(D_list, dtype=np.float32)
    R = np.array(R_raw, dtype=np.float32)

    if R.max() - R.min() > 1e-12:
        Rn = (R - R.min()) / (R.max() - R.min())
    else:
        Rn = np.zeros_like(R)

    D_hat = combine_metrics(
        D=D, Rn=Rn, alpha=args.risk_alpha, mode=args.combine_mode
    )

    all_layer_ratio = D_hat.copy()

    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) *
                       (1.0 / (all_layer_ratio.max() - all_layer_ratio.min() + 1e-12) * args.Lamda * 2))
    
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)

    print("after adjustment", all_layer_ratio, "mean", np.mean(all_layer_ratio),
          "max", np.max(all_layer_ratio), "min", np.min(all_layer_ratio))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    print(all_layer_ratio)
    return all_layer_ratio.tolist()


def dlp_score_global(args, model, tokenizer, device=torch.device("cuda:0")):
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

    all_layer_metric = []
    all_layer_numel = []
    R_raw = []

    risk_k = args.risk_k
    risk_probe = args.risk_probe
    risk_decay = args.risk_decay
    total_nsamples = args.nsamples
    risk_nsamples = args.risk_nsamples
    probe_method = args.probe_method

    attention_mask = cache.get("attention_mask", None)
    position_ids = cache.get("position_ids", None)
    is_opt = "opt" in args.base_model.lower()

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_numel = torch.tensor([subset[name].weight.numel() for name in subset])
        all_layer_numel.append(layer_numel)

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
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name, wrapped_layers_ref):
            def tmp(_, inp, out):
                wrapped_layers_ref[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name, wrapped_layers)))
        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        for h in handles:
            h.remove()

        layer_wmetric = []

        print(f"get layer sp layer {i}")
        for name in subset:
            # print(f"get layer sp layer {i} name {name}")
            W = torch.abs(subset[name].weight.data).to(dtype=torch.float32)
            X = wrapped_layers[name].scaler_row.reshape((1, -1)).to(dtype=torch.float32)
            W_metric = W * torch.sqrt(X)

            layer_wmetric.append(W_metric)

        if args.layer == "dlpb":
            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
            attn_numel = torch.sum(layer_numel[:4]).item()
            attn_median = torch.mean(layer_wmetric[:attn_numel])
            ffn_median = torch.mean(layer_wmetric[attn_numel:])
            layer_metric = torch.tensor([attn_median] * 4 + [ffn_median] * (len(layer_numel) - 4))
            all_layer_metric.append(layer_metric)
        elif args.layer == "dlpc":
            layer_metric = torch.tensor([torch.median(x).cpu().item() for x in layer_wmetric])
            all_layer_metric.append(layer_metric)

        j_end = min(len(layers) - 1, i + risk_k)
        risk_sum = torch.tensor(0.0, device=inps.device)
        w_sum = 0.0
        current_risk_nsamples = min(risk_nsamples, total_nsamples, args.num_examples)

        for j in range(current_risk_nsamples):
            x0 = inps[j].unsqueeze(0)

            base_outs = _forward_range(
                layers, i, j_end, x0,
                attention_mask=attention_mask, position_ids=position_ids, is_opt=is_opt
            )

            ctx = get_perturb_context(
                layer=layers[i],
                method=probe_method,
                wrapped_layers=wrapped_layers,
                probe_ratio=risk_probe
            )

            with ctx:
                pert_outs = _forward_range(
                    layers, i, j_end, x0,
                    attention_mask=attention_mask, position_ids=position_ids, is_opt=is_opt
                )

            include_self = args.risk_include_self
            if include_self:
                start_t = 0
            else:
                start_t = 0 if len(base_outs) == 1 else 1

            for t in range(start_t, len(base_outs)):
                w = (risk_decay ** t)
                d = _rel_l2(base_outs[t], pert_outs[t])
                risk_sum += w * d
                w_sum += w

        R_i = 0.0 if w_sum == 0 else (risk_sum / w_sum).item()
        R_raw.append(R_i)

        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

        del wrapped_layers
        torch.cuda.empty_cache()

    all_layer_metric = torch.stack(all_layer_metric, dim=0)
    all_layer_numel = torch.stack(all_layer_numel, dim=0).to(dtype=torch.float32)

    D = all_layer_metric.cpu().numpy().astype(np.float32)
    R = np.array(R_raw, dtype=np.float32)
    if R.max() - R.min() > 1e-12:
        Rn = (R - R.min()) / (R.max() - R.min())
    else:
        Rn = np.zeros_like(R)

    D_hat = combine_metrics(
        D=D,
        Rn=Rn.reshape(-1, 1),
        alpha=args.risk_alpha,
        mode=args.combine_mode,
    )

    D_hat = torch.from_numpy(D_hat).to(dtype=torch.float32)
    layer_imp = 1 - D_hat / torch.sum(D_hat)
    layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min() + 1e-12) * args.Lamda * 2
    print(layer_imp)
    layer_prune_numel = all_layer_numel * args.final_s + (torch.mean(layer_imp) - layer_imp) * torch.mean(
        all_layer_numel.float())
    all_layer_ratio = layer_prune_numel / all_layer_numel

    print(all_layer_ratio)
    print(torch.min(all_layer_ratio), torch.max(all_layer_ratio))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    return all_layer_ratio.tolist()


class DLP:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = [2, 0.06, 0.02, 0.04, 0.02, 0.04, 0.1, 0.15, 0.12] # zero index for n:m

    @torch.no_grad()
    def get_layer_sp(self, args):
        self.alpha[0] = args.Lamda
        if args.layer == "dlp":
            return dlp_score(args, self.model, self.tokenizer, self.alpha[int(args.final_s * 10)], device=self.model.device)
        elif args.layer == "dlpb" or args.layer == "dlpc":
            return dlp_score_global(args, self.model, self.tokenizer, device=self.model.device)