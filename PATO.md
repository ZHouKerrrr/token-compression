# PATO: Pareto‑Efficient VLMs via Conditional Precompression and Prefix‑Optimal Token Orders

---

# 1. 引言

## 1.1 背景

多模态大模型（MLLM）将图像编码为视觉 token 并与文本 token 融合驱动 LLM。传统“**事后压缩**”在视觉编码器之后再剪枝/合并视觉 token：
(1) 无法降低视觉端自注意力的二次复杂度；
(2) 未充分使用查询文本 ($X_q$) 的先验以做**任务相关的视野选择**；
(3) 常以注意力权重为代理，**与最终语言损失不对齐**。

## 1.2 核心思路

本文提出“双信息浓缩”的token压缩框架PATO：首先是一条**像素域的信息浓缩**管线：先在像素域通过**文本条件化**的可微映射 ($g_{\text{raw}}(X_v,X_q)$) 生成一张**浓缩小图** ($X_v^{*}$)（非为了人眼可视，仅为下游编码器“可理解”），再送入标准视觉编码器。随后用**Token Sort**按**边际收益**对视觉 token 排序，使**任意预算 ($M$)** 的**前缀**尽可能好。训练采用**随机预算**与**多预算蒸馏**，使目标直接与**最终语言损失**对齐。

## 1.3 本文档摘要

*   形式化一个**文本条件化的双预算**（像素/patch 与 token）优化问题；
*   给出高层计算图与损失；
*   详述 ($g_{\text{raw}}$) 的五个像素域方案（**I/C/F/A/B**）与 Token Sort 的三种方案（**A/B/C**），并给出对比与实现细节；
*   提供可方案原理的伪代码

---

# 2. 问题建模

## 2.1 统一记号

*   图像 ($X_v \in \mathbb{R}^{H\times W\times 3}$)，文本 ($X_q$)；
*   预压缩（像素域）：$X_v^{*}=g_{\text{raw}}(X_v,X_q;\phi_{\text{raw}})\in\mathbb{R}^{H^{*}\times W^{*}\times 3}$；
*   视觉编码器：$Z_v=E_\psi(X_v^{*})=[z^1,\ldots,z^N], z^i\in\mathbb{R}^{d_v}$；
*   Token 排序/选择：$Z_v^{\downarrow}=\Pi_\varphi(Z_v,X_q), Z_v'=[z^{\downarrow1},\ldots,z^{\downarrow M}]$；
*   投影：$H_v=W(Z_v')\in\mathbb{R}^{M\times d_h}$，文本嵌入：$H_q=T(X_q)$；
*   语言模型：$Y_a\sim f_\theta([H_q,H_v])$。

## 2.2 目标：条件信息瓶颈 + 双预算

在数据分布 ($\mathcal{D}$) 上，最小化语言损失并受像素/patch 与 token 预算约束：
$$
\min_{\phi_{\text{raw}},\psi,\varphi,W,\theta}\
\mathbb{E}_{(X_v,X_q,Y)\sim\mathcal{D}}\ \mathbb{E}_{M\sim\mathcal{D}_M}\Big[
\mathcal{L}_{\mathrm{LM}}\big(Y;\ f_\theta(T(X_q),W(\Pi_\varphi(E_\psi(g_{\text{raw}}(X_v,X_q)))_{[:M]}))\big)\Big]
+\lambda_{\text{pix}} \mathbb{E}[C_{\text{pix}}(X_v^{*})]
+\lambda_{\text{tok}} \mathbb{E}[M]
+\mathcal{R}.
$$
($\mathcal{R}$) 包含平滑/去冗余/覆盖等正则；($\mathcal{D}_M$) 为训练时的**随机预算**分布。

## 2.3 前缀最优（listwise 排序）

记损失 ($\mathcal{L}(k,\pi)$) 为排序 ($\pi$) 的前 ($k$) 个 token 的语言损失，Token Sort 目标为
$$
\min_{\pi=\pi_\varphi}\ \mathbb{E}_{k\sim\mathcal{D}_M}\ \mathbb{E}_{(X_v,X_q,Y)}\ \mathcal{L}(k,\pi).
$$
当效用呈次模性时，按边际收益排序近似最优（贪心 ($1-1/e$) 近似）。

---

# 3. 算法 High‑level pipeline

**输入**：图像 ($X_v$)、文本 ($X_q$)
**输出**：文本回答 ($Y_a$)

**Step‑1：像素域预压缩（文本条件化）**
$$
X_v^{*}=g_{\text{raw}}(X_v,X_q;\phi_{\text{raw}})\quad (\text{浓缩小图，减小输入图像的尺寸，非人眼可视导向})
$$

**Step‑2：视觉编码**
$$
Z_v=E_\psi(X_v^{*})=[z^1,\dots,z^N]
$$

**Step‑3：Token Sort（前缀最优）**
$$
Z_v^{\downarrow}=\Pi_\varphi(Z_v,X_q),\quad Z_v' = Z_v^{\downarrow}[1{:}M]
$$

**Step‑4：映射与融合**
$$
H_v=W(Z_v'),\quad H_q=T(X_q),\quad Y_a\sim f_\theta([H_q,H_v])
$$

**训练要点**

*   **随机预算** ($M$) 与像素预算（如瓦片数 ($K$)、核带宽等）；
*   **多预算蒸馏**（大预算 teacher → 小预算 student）；
*   **文本引导正则**：query‑swap 对比、分布蒸馏、平滑/覆盖。

**计算图（文字）**
$X_v \xrightarrow{g_{\text{raw}}(X_q)} X_v^{*} \xrightarrow{E_\psi} Z_v \xrightarrow{\Pi_\varphi(X_q)} Z_v' \xrightarrow{W} H_v \xrightarrow{\text{concat with } H_q} f_\theta \to Y_a$.

---

# 4. 算法实现详细的 pipeline

## 4.1 ($g_{\text{raw}}$)：像素域方案（I / C / F / A / B）

> 目标：在像素域生成 ($X_v^{*}$)，最大化与 ($X_q$) 相关的信息密度，最小化无关冗余，且**与标准视觉前端完全兼容**。

### 4.1.1 方案 I — 分块选择（Tiled Select‑and‑Place）

**机制**：固定或多尺度网格切块，打分 ($a_b=r_\phi(F_b,e_q)$)，soft‑Top‑($K$) 选块；每块经双线性重采样**按原相对位置**映射到小画布，边界羽化合成；叠加一张低分辨率全局底图 ($C_0$)。

**像素合成：**
$$
X_v^*(u,v)=\frac{\alpha_0 C_0(u,v)+\sum_{b\in\mathcal{S}}\alpha_b(u,v) C_b(u,v)}{\alpha_0+\sum_{b\in\mathcal{S}}\alpha_b(u,v)+\varepsilon}.
$$

**正则/预算**：块重叠+羽化（4–8 px）、最小间距/多样性；$\sum \tilde p_b \le K_{\max}$。
**优点**：TC 强、覆盖强、几何一致、稳定、工程简单。
**典型失败**：接缝伪影（羽化+重叠），尺度错配（多尺度网格）。

---

### 4.1.2 方案 C — 动态核下采样（Content‑Adaptive Kernel）

**机制**：输出像素 ($(u,v)$) 处预测核 ($w_{u,v}$)（由图像+文本条件化），对输入邻域加权：
$$
X_v^*(u,v)=\sum_{i,j} w_{u,v}(i,j),X_v(x(u,v,i),y(u,v,j)),\quad
w_{u,v}=\mathrm{softmax}\big(\mathrm{MLP}([F_{\text{local}}(u,v);\ e_q])\big).
$$
可做**基核混合**：$w_{u,v}=\sum_m \pi_{u,v,m}k_m$。

**正则/预算**：核归一化、带限约束（抑振铃）、($\mathrm{TV}(\pi)$) 平滑；像素预算通过输出分辨率控制。
**优点**：细节保真强（文档/边缘/文本）。
**典型失败**：核不稳（温度/幅度约束）与局部过锐化（带限正则）。

---

### 4.1.3 方案 F — 高斯涂抹（Gaussian Splatting in 2D）

**机制**：从显著图选 ($K$) 个中心（soft‑Top‑K），为每个中心生成二维高斯并写入画布：
$$
X_v^*(u,v)=\frac{\sum_{k=1}^{K}\alpha_k \mathcal{N}((u,v);\mu_k,\Sigma_k) c_k}{\sum_{k=1}^{K}\alpha_k \mathcal{N}(\cdot)+\varepsilon}.
$$
($c_k$) 为原图邻域加权颜色，($\mu_k,\Sigma_k,\alpha_k$) 由文本条件化网络预测。

**正则/预算**：中心 repulsion、多尺度带宽；($K$) 即像素预算代理。
**优点**：极低预算友好，易实现。
**典型失败**：($K$) 太小漏细节（与全局底图或 A 组合）。

---

### 4.1.4 方案 A — 加权下采样（Weighted Downsampling）

**机制**：学习显著密度 ($m(x,y|e_q)\in[0,1]$)，做归一化的抗混叠下采样：
$$
X_v^{*}(u,v)=\frac{\sum_{(x,y)\in\Omega(u,v)} m(x,y) \kappa(x,y;u,v) X_v(x,y)}
{\sum_{(x,y)\in\Omega(u,v)} m(x,y) \kappa(x,y;u,v)+\varepsilon}.
$$
**正则/预算**：($|m|_1$)（面积）、($\mathrm{TV}(m)$)（平滑）；
**优点**：最稳、最兼容；**强 baseline**。
**典型失败**：稀疏但分散（TV+面积下限）。

---

### 4.1.5 方案 B — 多尺度混合（Mixture‑of‑Scales Averaging）

**机制**：预生成多尺度图 ($\{X_v^{(s)}\}$)，逐像素学习非负权 ($\alpha_s(u,v|e_q)$) 做归一化混合：
$$
X_v^{*}(u,v)=\frac{\sum_{s}\alpha_s(u,v) X_v^{(s)}(u,v)}{\sum_s \alpha_s(u,v)+\varepsilon}.
$$
**正则/预算**：权重熵/温度；最高尺度支路可叠加 A 做精读。
**优点**：稳定、兼容；常优于固定下采样。
**典型失败**：单尺度偏置（熵/温度调度）。

---

### 4.1.6 文本引导力与分布桥接（通用）

*   **Query‑swap 对比**：对同一 ($X_v$) 换无关问题 ($X_q'$)，最大化 ($E_\psi(X_v^*(X_q))$) 与 ($E_\psi(X_v^*(X_q'))$) 的距离；
*   **特征蒸馏**：($\sum_{\ell}|E_\psi^{(\ell)}(X_v^{*})-E_\psi^{(\ell)}(\text{down}(X_v))|_2^2$)，桥接分布；
*   **像素轻约束**：限制色彩/对比度到预训练分布。

---

## 4.2 Token Sort：A / B / C

### 4.2.1 方案 A — 可微排序（SoftPerm/NeuralSort/Sinkhorn）

**打分器**：$s_i=r_\varphi(z^i,e_q,\bar z)$。
**近似置换**：$P=\mathrm{SoftPerm}(s;\tau)$，$Z_v^{\downarrow}=P^\top Z_v$。
**随机预算**：$k\sim\mathcal{D}_M$，前缀 $Z_v' = Z_v^{\downarrow}[1{:}k]$。
**正则**：前缀去冗余、排序熵、温度退火。
**优点**：与“边际收益排序”直接对齐。

### 4.2.2 方案 B — 随机门控（Hard‑Concrete / L0）

为每个 token 学保留概率 ($p_i$)，采样门 ($g_i$)（直通近似），期望目标：
$$
\mathcal{L}=\mathbb{E}_{g}\big[\mathcal{L}_{\text{LM}}(g\odot Z_v)\big]+\lambda_{0}\sum_i \mathbb{E}[g_i].
$$
推理按 ($p_i$) 排序取前 ($M$) 或阈值筛选。
**优点**：与“包含/不包含”的损失差（边际贡献）直接对齐。

### 4.2.3 方案 C — 多预算蒸馏（Teacher‑Student）

大预算 ($k^*$) 的 teacher 分布蒸馏到小预算 ($k$) 的 student：
$$
\mathcal{L}_{\text{KD}}=\mathrm{KL}\big(\mathrm{soft}(p_{k^*};T)\parallel \mathrm{soft}(p_{k};T)\big).
$$
**作用**：小预算性能更稳、前缀单调性更好。可与 A/B 联合。

---

# 5. 伪代码（可直接落地）

> 记：`SoftPerm` 为可微排序；`HardConcrete` 为门控；`KD` 为蒸馏；`alpha_blend` 为羽化加权合成。代码为框架级示意。

## 5.1 训练主循环（集成）

```python
# ---- 前向 ----
Xv_star = g_raw(Xv, Xq, mode=cfg.g_mode)     # 'I' | 'C' | 'F' | 'A' | 'B'
Zv       = E_psi(Xv_star)                    # [N, d_v]
Hq, eq   = T(Xq), TextProj(T(Xq))

# Token Sort
if cfg.sort == 'A':  # 可微排序
    scores = Ranker(Zv, eq)                  # [N]
    P = SoftPerm(scores, tau)                # [N, N]
    Z_sorted = P.T @ Zv
    k = sample_budget(M_min, M_max)
    Z_prefix = Z_sorted[:k]
elif cfg.sort == 'B': # 门控
    p_keep = Gate(Zv, eq)                    # [N] in (0,1)
    g = HardConcrete(p_keep, beta)           # straight-through
    Z_prefix = g[:, None] * Zv
else:  # 'C' 仅蒸馏需与 A/B 联用，这里示例 A
    scores = Ranker(Zv, eq); P = SoftPerm(scores, tau)
    k = sample_budget(M_min, M_max)
    Z_prefix = (P.T @ Zv)[:k]

Hv = W(Z_prefix)
logits = f_theta([Hq, Hv])
loss_main = CE(logits, Y)

# KD（可选）
loss_kd = 0.0
if cfg.use_kd:
    k_star = cfg.k_teacher
    Z_teacher = (P.T @ Zv)[:k_star]
    logits_t = stopgrad(f_theta([Hq, W(Z_teacher)]))
    loss_kd = KL(soft(logits_t, T_kd), soft(logits, T_kd))

# 文本引导力（query-swap）与蒸馏（桥接分布）
Xv_star_swap = g_raw(Xv, Xq_swap, mode=cfg.g_mode)
contrast = F.relu(margin - l2(E_psi(Xv_star), E_psi(Xv_star_swap)))
distill  = sum_l2(inter_feats(E_psi, Xv_star), inter_feats(E_psi, downsample_baseline(Xv)))

# 预算与正则
loss_reg = budget_pix_reg(...) + smoothing_reg(...) + coverage_reg(...)
if cfg.sort == 'A': loss_reg += ent_perm(P) + diversity(Z_prefix)
if cfg.sort == 'B': loss_reg += lambda_l0 * p_keep.sum()

loss = loss_main + beta_kd*loss_kd + lambda_contrast*contrast + lambda_distill*distill + loss_reg
loss.backward(); optimizer.step(); scheduler.step()
```

## 5.2 ($g_{\text{raw}}$) — 方案 I：分块选择（多尺度 + 羽化）

```python
def g_raw_I(Xv, Xq, target_size=(448,448), K_total=8, scales=(1.0, 0.5, 0.25)):
    eq = TextProj(T(Xq))
    cand = []  # (scale, bbox_xyxy, score)
    for s in scales:
        Xs = resize(Xv, scale=s)
        Fs = LightCNN(Xs)
        for b in grid_tiles(Xs, tile_size=T, overlap=ov):
            a = score_tile(Fs[b], eq)        # [1]
            cand.append((s, b.bbox, a))
    idx = soft_topk_indices([a for (_,_,a) in cand], K_total, tau_sel)
    selected = [cand[i] for i in idx]

    canvas = resize(Xv, target_size); alpha = alpha_const(target_size, val=alpha0)
    for (s, bbox, _) in selected:
        crop = grid_sample_from_bbox(Xv, bbox, dst=map_to_canvas(bbox, target_size))
        a_m  = feather_mask_like(crop, feather_px=Fpx)
        canvas = canvas + a_m * crop
        alpha  = alpha  + a_m
    Xv_star = canvas / (alpha + 1e-6)
    return Xv_star
```

## 5.3 ($g_{\text{raw}}$) — 方案 C：动态核下采样（基核混合）

```python
def g_raw_C(Xv, Xq, target_size=(448,448), kernels=Kbank):  # Kbank: {k_m}
    eq = TextProj(T(Xq))
    F  = LightCNN(Xv)
    Hs, Ws = target_size
    # 局部特征映射到目标网格
    Floc = sample_local_features(F, Hs, Ws)                  # [Hs, Ws, d]
    pi   = softmax(MLP(concat([Floc, expand(eq, Hs, Ws)])), dim=-1)  # [Hs, Ws, M]
    # im2col + 基核混合卷积
    patches = unfold_neighborhood(Xv, Hs, Ws, ksize=k)       # [Hs, Ws, k*k, 3]
    mixed_k = (pi[...,None] * Kbank_weights).sum(dim=-2)     # [Hs, Ws, k*k]
    Xv_star = (mixed_k[...,None] * patches).sum(dim=-2)      # [Hs, Ws, 3]
    return Xv_star
```

## 5.4 ($g_{\text{raw}}$) — 方案 F：高斯涂抹

```python
def g_raw_F(Xv, Xq, target_size=(448,448), K=8):
    eq  = TextProj(T(Xq))
    Fg  = GlobalFeat(Xv)
    centers, covs, amps = select_gaussians(Fg, eq, K, tau=tau_sel)   # soft-top-k
    colors  = sample_colors(Xv, centers, covs)                       # 邻域加权均值
    U, V = meshgrid(target_size)
    num   = 0; den = 1e-6
    for k in range(K):
        G = gaussian(U,V, centers[k], covs[k])                       # [Hs, Ws, 1]
        num += amps[k]*G*colors[k];  den += amps[k]*G
    Xv_star = num / den
    return Xv_star
```

## 5.5 ($g_{\text{raw}}$) — 方案 A：加权下采样（抗混叠）

```python
def g_raw_A(Xv, Xq, target_size=(448,448)):
    F  = LightCNN(Xv)
    eq = TextProj(T(Xq))
    m  = sigmoid(conv1x1(FiLM(F, eq)))                               # [H, W, 1]
    Xv_star = normalized_weighted_downsample(Xv, m, target_size, kernel='bilinear')
    return Xv_star
```

## 5.6 ($g_{\text{raw}}$) — 方案 B：多尺度混合

```python
def g_raw_B(Xv, Xq, target_size=(448,448), scales=(1.0, 0.5, 0.25)):
    eq = TextProj(T(Xq))
    pyr = [resize(Xv, scale=s, dst=target_size) for s in scales]     # 同尺度画布
    F   = [LightCNN(p) for p in pyr]
    # 逐像素权重
    alpha = [softplus(MLP(concat([Fi, expand(eq, *target_size)]))) for Fi in F]
    denom = sum(alpha) + 1e-6
    Xv_star = sum([a*p for a,p in zip(alpha, pyr)]) / denom
    return Xv_star
```

## 5.7 Token Sort — 方案 A：可微排序

```python
def token_sort_A(Zv, Xq, tau):
    eq = TextProj(T(Xq))
    s  = Ranker_phi(concat([Zv, expand(eq, len(Zv))]))   # [N, 1] -> [N]
    P  = soft_permutation(s, tau)                        # [N, N]
    Z_sorted = P.T @ Zv
    return Z_sorted, P
```

## 5.8 Token Sort — 方案 B：门控（Hard‑Concrete）

```python
def token_sort_B(Zv, Xq, beta):
    eq = TextProj(T(Xq))
    p_keep = sigmoid(MLP(concat([Zv, expand(eq, len(Zv))])))  # [N]
    g = hard_concrete_sample(p_keep, beta)                    # straight-through
    Z_masked = g[:, None] * Zv
    return Z_masked, p_keep
```

## 5.9 Token Sort — 方案 C：多预算蒸馏（包装器）

```python
def kd_loss(student_logits, teacher_logits, T):
    return KL(soft(teacher_logits, T), soft(student_logits, T))
```

---

### 训练日程（建议）

*   **Stage‑0**（稳定期，2–5k step）：冻结 ($E_\psi$)，仅训 ($g_{\text{raw}}$)+排序器；排序温度/门温度较高。
*   **Stage‑1**（联合期，20–80k step）：解冻 ($E_\psi$) 后半层；启用随机预算（像素/Token）。
*   **Stage‑2**（收敛期）：启用**多预算蒸馏**；退火排序温度/门温度到推理状态；针对部署预算作再加权。

### 关键超参（默认区间）

*   输出分辨率 ($H^{*} \times W^{*}$)：如 $448^2$；
*   方案 I：每尺度 ($K_s=2\sim 8$)、重叠 10–20%、羽化 4–8 px；
*   方案 C：核 3×3/5×5，基核 ($M=4\sim 8$)，带限正则 ($10^{-4}\sim10^{-3}$)；
*   方案 F：($K=4\sim 16$)、带宽多尺度、repulsion 权 ($10^{-3}\sim10^{-2}$)；
*   方案 A：($\lambda_{\text{pix}}=10^{-4}\sim10^{-3}$)、($\lambda_{\text{TV}}=5\times10^{-5}\sim5\times10^{-4}$)；
*   方案 B：温度 ($\tau_{\text{scale}}=1.0\to0.3$) 退火；
*   排序 A：($\tau=2.0\to0.3$)，去冗余权 ($10^{-3}\sim10^{-2}$)；
*   门控 B：($\beta=5.0\to1.0$)、($\lambda_{0}$) 使 ($\sum \mathbb{E}[g_i]\approx \bar M$)；
*   蒸馏 C：温度 ($T=2$)、权 ($\beta_{\text{kd}}=0.5$)。

*   **Stage‑0**（稳定期，2–5k step）：冻结 ($E_\psi$)，仅训 ($g_{\text{raw}}$)+排序器；排序温度/门温度较高。
*   **Stage‑1**（联合期，20–80k step）：解冻 ($E_\psi$) 后半层；启用随机预算（像素/Token）。
*   **Stage‑2**（收敛期）：启用**多预算蒸馏**；退火排序温度/门温度到推理状态；针对部署预算作再加权。

### 关键超参（默认区间）

本研究将**文本条件化的像素域浓缩**作为门户，统一在**双预算**框架下与**前缀最优排序**协同优化。所选 ($g_{\text{raw}}$) 五个方案（I/C/F/A/B）覆盖从“覆盖稳健”到“细节上限”再到“极低预算”的多种需求，Token Sort（A/B/C）提供“任意预算可用”的前缀质量保障。所有模块均**端到端可微**、与现有视觉前端**完全兼容**，并以**最终语言损失**为导向而非注意力代理。
