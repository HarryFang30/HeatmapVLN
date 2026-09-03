# 3 方法

本节首先形式化流式导航任务及其输出（Sec. 3.1），随后介绍连续图像流上的历史位姿估计（Sec. 3.2）和四视角热力图预测（Sec. 3.3），最后说明原生 InternNav 的局部 waypoint 生成过程与训练目标（Sec. 3.4-3.5）。本文方法包含两个并行输出：空间分支预测四视角历史热力图，动作分支预测由 32 个 waypoint 组成的局部轨迹；两者分别描述“历史空间在哪里”和“接下来如何移动”。

## 3.1 流式导航问题定义

给定自然语言导航指令 \(q\) 和按时间到达的连续前视图像流

\[
\mathcal L=\{I_1,I_2,\ldots\},\qquad
\mathcal L_t=\{I_1,\ldots,I_t\},\qquad
I_i\in\mathbb R^{H\times W\times3},
\]

系统在时刻 \(t\) 只能访问当前帧及已经观测到的历史帧 \(\mathcal L_t\)，不能使用任何未来图像 \(I_{t+j},j>0\)。这里的“连续”描述环境观测按时间不断到达；具体网络可以从该前缀中选择有限历史帧，但任何选择都必须满足因果约束。我们希望从同一流式视觉输入得到两类互补输出：

\[
\mathcal F(q,\mathcal L_t)
=\left(\widehat{\mathcal H}_t,\widehat{\mathcal W}_t\right).
\]

其中，

\[
\widehat{\mathcal H}_t
=\left\{\widehat H_{t\leftarrow\tau_k}^{d}\right\}
_{k=1,\ldots,K;\,d\in\mathcal D},
\quad
\mathcal D=\{\mathrm F,\mathrm R,\mathrm B,\mathrm L\},
\]

表示 \(K\leq8\) 个历史位置在当前 Front、Right、Back、Left 四个相机方向中的空间分布，且

\[
\widehat{\mathcal H}_t\in[0,1]^{K\times4\times64\times64}.
\]

第二个输出

\[
\widehat{\mathcal W}_t
=\{\widehat{\mathbf w}_{t,n}\}_{n=1}^{32}
\in\mathbb R^{32\times3}
\]

是原生 InternNav 动作分支生成的 32 步局部 waypoint 序列。于是，固定指令 \(q\) 时，整个系统可以简写为

\[
\mathcal L_t
\longrightarrow
\begin{cases}
\widehat{\mathcal H}_t,&\text{四视角历史热力图},\\
\widehat{\mathcal W}_t,&\text{32-waypoint 局部轨迹}.
\end{cases}
\]

为了区分“连续输入源”和“网络实际取用的帧”，我们分别记热力图与动作分支的因果采样算子为 \(\mathcal S_{\rm hm}\) 和 \(\mathcal S_{\rm act}\)。前者为 Heatmap Head 选择当前帧和至多八张历史帧；后者保留 InternNav 原生的历史观测组织方式。两条分支共享已经到达的图像流，但在当前方法中相互独立：

\[
\widehat{\mathcal H}_t
=\mathcal G_{\rm hm}\!\left(\mathcal S_{\rm hm}(\mathcal L_t)\right),
\qquad
\widehat{\mathcal W}_t
=\mathcal G_{\rm act}\!\left(q,\mathcal S_{\rm act}(\mathcal L_t)\right),
\]

而不是令 \(\widehat{\mathcal W}_t\) 以 \(\widehat{\mathcal H}_t\) 为条件。

## 3.2 从连续图像流估计历史位姿

**连续视觉里程计。** 热力图需要回答“过去的位置现在位于哪个方向、哪个像素”，因此首先需要恢复历史相机与当前相机之间的空间关系。我们采用冻结的 AMB3R-VO 作为流式位姿提供器。其作用可抽象为

\[
\mathcal V_{\rm vo}:\mathcal L_t
\longmapsto
\widehat{\mathcal P}_{1:t}
=\{\widehat{\mathbf T}_1,\ldots,\widehat{\mathbf T}_t\},
\qquad
\widehat{\mathbf T}_i\in\mathrm{SE}(3).
\]

AMB3R 接收的是按顺序到达的连续 RGB 帧，而不是 Heatmap Head 最终使用的 \(K\) 张稀疏历史图。它维护随观测更新的相机轨迹，并输出各帧的 camera-to-world 位姿。这里我们只把 AMB3R 视为从 RGB 流到历史位姿流的冻结、因果映射，不展开其内部建图或匹配过程，也不在导航模型中重新训练它。

**历史选择与相对表示。** 在时刻 \(t\)，从已观测帧中均匀选取

\[
\Gamma_t=\{\tau_1,\ldots,\tau_K\},
\qquad 1\leq\tau_1<\cdots<\tau_K<t,
\]

并以当前相机为参考计算

\[
\widehat{\mathbf T}_{t\leftarrow\tau_k}
=\widehat{\mathbf T}_t^{-1}\widehat{\mathbf T}_{\tau_k}.
\]

在统一相机坐标约定后，我们保留平面前向位移、左向位移和相对航向角：

\[
\widehat{\mathbf p}_{t\leftarrow\tau_k}
=\left[
\Delta\widehat f_k,
\Delta\widehat l_k,
\cos\Delta\widehat\psi_k,
\sin\Delta\widehat\psi_k
\right]\in\mathbb R^4.
\]

因此，空间分支的几何输入流可以写成

\[
\mathcal L_t
\xrightarrow{\mathcal V_{\rm vo}}
\widehat{\mathcal P}_{1:t}
\xrightarrow{\rm relative\ pose}
\widehat{\mathbf P}_t
\in\mathbb R^{K\times4}.
\]

训练和部署均使用这一预测位姿表示作为 Head 输入；GT 位姿不进入该输入流。

## 3.3 四视角热力图预测

**视觉表征。** 对当前帧 \(I_t\) 和被选中的历史帧 \(\{I_{\tau_k}\}_{k=1}^{K}\)，冻结但实际执行的 Qwen.visual 提取视觉特征：

\[
\mathcal E_{\rm vis}
\left(I_t,\{I_{\tau_k}\}_{k=1}^{K}\right)
=\left(\mathbf S_t,\{\mathbf h_k\}_{k=1}^{K}\right).
\]

\(\mathbf S_t\) 保留当前前视图的空间布局，\(\mathbf h_k\) 是第 \(k\) 张历史图的池化视觉表示。该热力图分支只调用 Qwen 的视觉编码器，不调用语言 Transformer。这里“冻结”不等于“不运行”：Qwen.visual 在每次前向中真实提取特征，只是不参与参数更新；动作分支则保持原生 InternNav。由于热力图模型输入只有前视 RGB，Front 槽由真实当前空间特征构成；Right、Back、Left 槽由可学习方向查询、方向编码以及前视全局上下文构成。四视角输出是空间预测结果，并不意味着模型额外读取了三张侧后 RGB。

**方向空间构造。** 当前帧的四个规范方向槽可进一步写为

\[
\mathbf S_t^{\mathrm F}=\mathbf S_t,\qquad
\mathbf S_t^d
=\mathbf Q^d+\mathbf e^d+
\gamma^d\!\left(\operatorname{GAP}(\mathbf S_t)\right),
\quad d\in\{\mathrm R,\mathrm B,\mathrm L\},
\]

其中 \(\mathbf Q^d\) 是每个方向独立学习的 canonical spatial query，\(\mathbf e^d\) 编码相对 Front 的固定方位角，\(\gamma^d\) 将当前前视图的全局上下文调制到对应方向。该构造并不试图合成侧后 RGB；它只建立四个具有明确方向语义的空间承载槽，使历史点能够被分类到正确视角并在该视角内定位。于是，Heatmap Head 的完整输入不是单独的位姿或单独的图像，而是

\[
\mathcal X_t^{\rm hm}
=\left(
I_t,\{I_{\tau_k}\}_{k=1}^{K},
\{\widehat{\mathbf p}_{t\leftarrow\tau_k}\}_{k=1}^{K}
\right).
\]

**位姿编码。** 对每个历史相对位姿，使用原有 Fourier 映射

\[
\mathbf r_k=\mathcal E_{\rm pose}
\left(\widehat{\mathbf p}_{t\leftarrow\tau_k}\right)\in\mathbb R^{256}.
\]

具体地，四维位姿先归一化，再与 16 组正弦、余弦频率拼接为 132 维表示，随后由线性层投影为 trajectory token。它与历史视觉 token 和当前四方向空间 token 联合输入两层 Transformer：

\[
\mathbf z_k^h=\mathbf W_h\mathbf h_k\in\mathbb R^{256},
\qquad
\mathbf Z_k=\mathcal T_{\rm hm}\left(
[\mathbf z_k^h;\mathbf r_k;
\mathbf S_t^{\mathrm F},
\mathbf S_t^{\mathrm R},
\mathbf S_t^{\mathrm B},
\mathbf S_t^{\mathrm L}]
\right).
\]

这里，历史视觉 token 提供“这个历史观测是什么”，trajectory token 提供“它相对当前相机在哪里”，四方向空间 token 提供“预测应落在哪个当前视角及其像素位置”。三类信息在同一个历史槽内融合，但不同历史槽仍保持独立索引，因此输出可以同时表达多个过去位置，而不会丢失各自的时间身份。

**四视角输出。** 轨迹引导模块先预测每个历史点在四个方向上的 visibility logits

\[
\boldsymbol\ell_t\in\mathbb R^{K\times4},
\qquad
\widehat{\mathbf v}_t=\sigma(\boldsymbol\ell_t),
\]

以及 coarse heatmap

\[
\widehat{\mathcal H}^{\rm c}_t
\in\mathbb R^{K\times4\times8\times8}.
\]

Fine Decoder 再结合当前空间特征将其细化为四视角像素 logits：

\[
\mathbf A_t
=\mathcal G_{\rm hm}
\left(
I_t,\{I_{\tau_k}\},
\{\widehat{\mathbf p}_{t\leftarrow\tau_k}\}
\right),
\qquad
\mathbf A_t
\in\mathbb R^{K\times4\times64\times64}.
\]

推理时在每个视角内做空间 softmax：

\[
\Pi_{t\leftarrow\tau_k}^{d}(u,v)
=\operatorname{softmax}_{u,v}
\left(A_{t\leftarrow\tau_k}^{d}\right)(u,v).
\]

因此，对任意 \(k\) 和 \(d\)，条件空间分布满足

\[
\sum_{u=1}^{64}\sum_{v=1}^{64}
\Pi_{t\leftarrow\tau_k}^{d}(u,v)=1.
\]

四个 visibility logits 与固定的 none logit \(0\) 共同构成五类分布

\[
\boldsymbol\alpha_k
=\operatorname{softmax}
\left([0,\ell_k^{\mathrm F},\ell_k^{\mathrm R},
\ell_k^{\mathrm B},\ell_k^{\mathrm L}]\right).
\]

最终四视角热力图为

\[
\widehat H_{t\leftarrow\tau_k}^{d}(u,v)
=\alpha_k^d\,
\Pi_{t\leftarrow\tau_k}^{d}(u,v),
\qquad d\in\mathcal D,
\]

并可用峰值位置

\[
(\widehat u_k^d,\widehat v_k^d)
=\underset{u,v}{\arg\max}\;
\widehat H_{t\leftarrow\tau_k}^{d}(u,v)
\]

表示预测的空间点。每张图因此对应一个历史相机中心投影到当前相机某个方向后的像素概率分布；额外的 none/四方向分类用于区分不可见历史点和 Front、Right、Back、Left 中的有效视角。最终输出不是把 \(K\) 个历史点压成一个无身份的响应图，而是保留“哪个历史点、位于哪个方向、落在哪个像素”的三层索引。由此，完整空间映射可概括为

\[
\boxed{
\mathcal L_t
\rightarrow
\widehat{\mathcal P}_{1:t}
\rightarrow
\widehat{\mathbf P}_t
\rightarrow
\left(\widehat{\mathcal H}_t,
\widehat{\mathbf v}_t\right)}.
\]

## 3.4 流式推理与导航输出

**热力图推理。** 当新帧 \(I_t\) 到达时，系统依次执行四个因果步骤：首先将 \(I_t\) 写入 AMB3R 的连续 RGB 状态并更新位姿序列；其次从已观测前缀中确定 \(\Gamma_t\)，将相应历史位姿变换到当前相机坐标系；随后用冻结的 Qwen.visual 编码当前帧和选中的历史帧；最后由 Heatmap Head 同时输出 \(K\) 个历史点的 visibility 与四方向像素分布。该过程可记为

\[
I_t
\xrightarrow{\rm ingest}
\widehat{\mathcal P}_{1:t}
\xrightarrow{\rm select/encode}
\mathcal X_t^{\rm hm}
\xrightarrow{\rm Heatmap\ Head}
(\widehat{\mathcal H}_t,\widehat{\mathbf v}_t).
\]

该推理流不会访问未来图像；GT pose、depth 和 intrinsics 也不属于部署输入。连续 RGB 负责维持视觉里程计状态，稀疏选择的 \(K\) 张图负责 Heatmap Head 的历史视觉匹配，二者具有不同的时间采样密度但来自同一观测前缀。

**导航输出。** 动作生成完全沿用原生 InternNav [InternNav]，不是本文需要重新展开的组件。我们仅保留其输入输出定义：在指令 \(q\) 和原生方式组织的历史、当前 RGB 观测条件下，经过 InternNav 原生的候选生成与聚合后，最终得到一条由 32 个稠密、固定间隔 waypoint 构成的局部连续轨迹，

\[
\widehat{\mathcal W}_t
=\mathcal G_{\rm InternNav}
\left(q,\mathcal S_{\rm act}(\mathcal L_t)\right)
=\left[\widehat{\mathbf w}_{t,1},\ldots,
\widehat{\mathbf w}_{t,32}\right]
\in\mathbb R^{32\times3}.
\]

其内部双系统结构、条件表示和轨迹生成目标均遵循 InternNav 原论文。本文关注的是与其并行输出的显式四视角空间预测。因此当前系统的整体输出为

\[
\boxed{
(q,\mathcal L_t)
\longrightarrow
\left(
\underbrace{\widehat{\mathcal H}_t}_{K\times4\times64\times64},
\underbrace{\widehat{\mathcal W}_t}_{32\times3}
\right)}.
\]

这里的联合输出表示同一时刻同时获得“历史空间在哪里”和“接下来如何移动”两类结果。从输入输出角度，两条路径可以分别概括为

\[
\mathcal L_t
\xrightarrow{\rm AMB3R+Heatmap\ Head}
\widehat{\mathcal H}_t,
\qquad
(q,\mathcal L_t)
\xrightarrow{\rm Native\ InternNav}
\widehat{\mathcal W}_t.
\]

## 3.5 热力图监督与位姿域适配

GT 相机轨迹仅用于生成训练标签：我们将每个历史相机中心投影至当前 Front、Right、Back、Left 相机平面，利用相机内参进行针孔投影，利用深度判断遮挡，并在可见像素处绘制高斯目标。记标签为 \(\mathcal H_t^*\) 和 \(\mathbf v_t^*\)，则训练样本的输入与监督关系为

\[
\underbrace{\left(
\mathcal L_t,
\widehat{\mathbf P}_t^{\rm AMB3R}
\right)}_{\text{模型输入}}
\longrightarrow
\underbrace{\left(
\mathcal H_t^*,\mathbf v_t^*
\right)}_{\text{GT 几何监督}}.
\]

我们从 GT-pose 训练得到的最佳 Head 初始化，并在训练时 100% 使用 AMB3R 预测位姿，使输入分布与部署一致。优化目标沿用原有可见性、像素定位、坐标回归、不可见抑制和方向分类损失：

\[
\mathcal L_{\rm hm}
=\mathcal L_{\rm vis}
+\mathcal L_{\rm spatial}
+0.2\mathcal L_{\rm coord}
+0.25\mathcal L_{\rm neg}
+0.5\mathcal L_{\rm spatial}^{\rm macro}
+\mathcal L_{\rm 5way}
+0.5\mathcal L_{\rm dir}^{\rm macro}.
\]

第一阶段仅微调 pose projection、两层融合 Transformer、visibility head 和 coarse heatmap head；Qwen.visual、当前图 DPT-Lite、历史视觉投影、方向 conditioner 与 Fine Decoder 均冻结但参与前向。原生 InternNav 动作分支保持不变，也不与 Heatmap Head 联合训练。该训练目标可概括为：在存在真实 VO 误差的历史位姿输入下，仍恢复由 GT 几何定义的四视角空间分布。

推理时不再读取用于制标签的 GT pose、depth 或 intrinsics。每个新 RGB 观测一方面更新 AMB3R 历史位姿流，另一方面进入 InternNav 的原生观测前缀；当系统触发规划时，同一因果前缀分别给出四视角热力图和 32-step waypoint 轨迹。由此，方法的输入始终可以归结为“导航指令加截至当前时刻的连续图像流”，输出则被完整定义为“显式历史空间分布加局部连续动作序列”。
