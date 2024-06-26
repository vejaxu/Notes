# 《机器学习》-ed1-周志华-学习笔记

## ch4-决策树

一颗决策树包含一个根结点，若干个内部结点，若干个叶结点。叶结点对应于决策结果，其他每个结点对应于一个属性测试。

决策树生成是一个递归的过程：

- 当前结点包含的样本全属于同一个类被，无需划分
- 当前属性集为空，或者所有样本在所有属性上取值相同，无法划分——当前结点标记为叶结点
- 当前结点包含的样本集合为空，不能划分

### 划分选择

信息熵（information entropy）是度量样本集合纯度最常用的一种指标，对于样本集合D中第k类样本所占比例为$p_k$ ，样本集合D的信息熵为：
$$
Ent\left(D\right) = - \sum_{i=1}^{|y|} p_k \log_2 p_k
$$
熵值越少，纯度越高。

假定离散属性 $\alpha$ 有V个可能的取值$\lbrace\alpha^1, \alpha^2, ..., \alpha^V \rbrace$ ，若使用$\alpha$​ 对样本D进行划分，产生V个分支结点。

- 信息增益（information gain）：

$$
Gain\left(D, \alpha \right) = Ent\left(D\right) - \sum_{v=1}^{V} \frac{|D^v|}{|D|}Ent(D^v)
$$

信息增益越大，属性 $\alpha$ 对样本集合划分所获得的纯度提升越大。

- 增益率（gain ratio）：

$$
Gain_ratio(D, \alpha) = \frac {Gain(D, \alpha)}{IV(\alpha)} \\
IV(\alpha) = - \sum _{v=1}^{V} \frac{|D_v|}{|D|} \log_2{\frac{|D_v|}{|D|}}
$$

其中，$IV(\alpha)$ 被称为属性 $\alpha$​ 的固有值。

- Gini指数

$$
Gini(D) = 1 - \sum_{k=1}^{|y|}p_k^2 \\
Gini\_index(D, \alpha) = \sum_{v=1}^{V} \frac{|D^v|}{|D|} \cdot Gini\left(D^v\right)
$$

Gini指数反应了从数据集D中随机抽取两个样本，其类别标记不一致的概率，指数越小，纯度越高。

### 剪枝处理

剪枝（pruning）是决策树学习算法对付“过拟合”的主要手段。

基本策略为预剪枝和后剪枝

- 预剪枝是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能的提升，则停止划分并将当前结点标记为叶结点。
- 后剪枝则是先从训练集生成一棵完整的决策树，然后自底向上地对非叶节点进行考察，若将该结点对应的子树替换为叶结点能够带来决策树泛化性能的提升，则将该子树替换为叶结点。

判断泛化性能提升的方法：使用留出法，预留一部分数据用作验证集进行性能评估。

### 连续与缺失值

当属性为连续值，可以使用连续属性离散化。在C4.5决策树算法中，采用二分法（bi-partition）对连续属性进行处理。

给定样本集D以及连续属性 $\alpha$ ，假定 $\alpha$ 在D上出现n个不同的取值，将n个值进行大小排序 $\lbrace\alpha^1, \alpha^2, ..., \alpha^n \rbrace$ 。基于划分点t可将D分为子集 $D_t^-$ 以及 $D_t^+$ 。其中左侧子集包含属性 $\alpha$ 取值不大于t的样本，右侧子集包含取值大于t的样本。

则候选划分点集合为
$$
T_\alpha = \lbrace \frac{\alpha^i + \alpha^{i + 1}}{2} | 1 \leq i \leq n-1 \rbrace
$$
则候选划分点集合有n-1个元素。

对每个候选点，其次计算二分后信息增益，选取信息增益最大的候选点作为划分点。

当样本的某些属性值缺失时，需要解决两个问题：

- 如何在属性值缺失的情况下进行划分属性选择？
- 给定划分属性，若样本在该属性上的值缺失，如何对该样本进行划分？

### 多变量决策树

将每个属性是为坐标空间中的一个坐标轴，则d个属性描述的样本对应d维空间中的一个数据点，对样本分类即表示在这个坐标空间寻找不同类样本之间的分类边界。

决策树形成的分类边界有一个明显的特点：轴平行。即分类边界由若干个与坐标轴平行的分段组成。

分类边界的每一段都与坐标轴平行，这样的分类边界使得学习结果有较好的可解释性，但当真是分类边界比较复杂时，必须使用很多段划分才能获得较好的近似。

而若使用斜的划分边界，模型得到大幅简化。多变量决策树实现斜的划分。

### 小结

决策树算法：ID3，C4.5，CART

特征选择：信息增益，增益率，基尼指数

多变量决策树：OC1——贪心地寻找每个属性的最优权值

增量学习（incremental learning）：接收到新样本后对已学得的模型进行调整

## ch5-神经网络

### 神经元

神经网络中最基本的成分是神经元模型，当某个神经元的电位超过一个“阈值”，则神经元会被激活。

M-P神经元模型：神经元接收来自n个其他神经元传递过来的输入信号，这些输入信号通过带权重的连接进行传递，神经元接收到的总输入值与神经元的阈值进行比较，然后通过激活函数处理以产生输出。

理想激活函数为阶跃函数，将输入映射为0或1
$$
sgn(x) = \begin{cases}
		1, &x \geqslant 0 \\
		0, &x < 0 
		\end{cases}
$$
但是阶跃函数不连续，不光滑

因而使用sigmoid函数作为激活函数
$$
sigmoid(x) = \frac{1}{1 + e^{-x}}
$$

### 感知机与多层网络

感知机（perceptron）由两层神经元组成，输入层接收外界输入信号后传递给输出层，输出层是M-P神经元，也称“阈值逻辑单元”。

## ch6-支持向量机

### 间隔与支持向量

SVM只要解决分类问题——二分类问题，基本想法在于基于训练集D在样本空间中找到一个划分超平面（能将样本分开的划分超平面很多，如何找到最优超平面？）

样本空间中，划分超平面可用如何方程表示：
$$
\omega^Tx + b = 0
$$
其中 $\omega = (\omega_1, \omega_2, ..., \omega_n)$ 为法向量，决定超平面的额方向，b为位移项，决定超平面与原点的距离。

样本空间中任意点到超平面的距离为：
$$
r = \frac{|\omega^Tx +b|}{||\omega||}
$$
如果超平面能够正确分类，则对于 $(x_i, y_i) \in D$ ，若 $y_i=+1$ ，则 $\omega^Tx + b > 0$ ，若 $y_i  = -1$ ， 则 $\omega^Tx+b<0$ 。

### SVM原始形式推导

要求分离超平面 $\omega^* \cdot x + b^* = 0$ 以及对应的决策函数 $f(x) = sign(\omega^* \cdot x + b^*)$

函数间隔： $\hat {r_i} = y_i (w \cdot x_i + b)$ 

则数据集D中所有样本距离超平面的函数间隔为： $\hat{r} = \min_{i=1,...,n}\hat{r_i}$

当成比例地改变 $\omega$ 与b时，超平面没有改变，但是函数间隔却发生改变，因而需要将法向量规范化得到几何间隔。

几何间隔： $r_i = \frac{\omega}{||\omega||} \cdot x_i + \frac{b}{||\omega||}$

几何间隔具有正负号，当 $y_i = 1$ 时，为正，因而几何间隔可以写为：
$$
r_i = y_i (\frac{\omega}{||\omega||} \cdot x_i + \frac{b}{||\omega||})
$$
样本集的几何间隔为
$$
r = \min_{i=1,...,n}r_i
$$
则函数间隔与几何间隔的关系：
$$
\gamma = \frac{\hat{\gamma}}{||\omega||}
$$
则SVM可得到最优化问题：
$$
\max_{\omega, b} \gamma \\
s.t. y_i(\frac{\omega}{||\omega||} \cdot x_i + \frac{b}{||\omega||}) \geqslant \gamma
$$
考虑到函数间隔与几何间隔的关系，可以转换为:
$$
\max_{\omega, b} \frac{\hat{r}}{||\omega||} \\
s.t. y_i(\omega \cdot x_i + b) \geqslant \hat{r}
$$
函数间隔 $\hat{r}$ 的取值不影响最优化问题的解，则可以将 $\hat{r} = 1$ ，且最大化 $\frac{1}{||\omega||}$ 与最小化 $\frac{1}{2} {||\omega||}^2$ 等价

则问题转换为：
$$
\min_{\omega, b} \frac{1}{2} {||\omega||}^2 \\
s.t. y_i(\omega \cdot x_i + b) \geqslant 1
$$
在线性可分情况下，数据集中的样本点中与分离超平面距离最近的点称为支持向量，支持向量是使约束条件成立的点。

对于正实例，支持向量在超平面 $\omega \cdot x + b = 1$

对于负实例，支持向量在超平面 $\omega \cdot x + b = -1$

二者中间的带宽为间隔，间隔大小为 $\frac{2}{||\omega||}$

### SVM对偶形式

使用拉格朗日对偶性，定义拉格朗日函数，为每个约束项添加一个拉格朗日乘子。
$$
L(\omega, b, \alpha) = 	\frac{1}{2}{||\omega||}^2 - \sum_{i=1}^{n}{\alpha_i}{y_i}(\omega \cdot x_i + b) + \sum_{i=1}^{n}{\alpha_i}
$$
原始问题的对偶问题为极大极小问题： $\max_{\alpha}\min_{\omega, b}L(\omega, b, \alpha)$

先求 $L(\omega, b, \alpha)$ 分别对 $\omega, b$ 的极小，再对 $\alpha$ 的极大

求得：
$$
\omega = \sum_{i=1}^{n} \alpha_i y_i x_i \\
0 = \sum_{i=1}^{n} \alpha_i y_i
$$
带回原式，得：
$$
\max_{\alpha} \lbrace\sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i\alpha_jy_iy_jx_i^Tx_j\rbrace \\
s.t. \sum_{i=1}^{n}\alpha_iy_i = 0 \\ \alpha_i \geqslant 0
$$
求解上式可以使用Sequential Minimal Optimization算法

### 核函数

在现实任务中，原始样本空间内也许并不存在一个能正确划分两类样本得超平面，比如异或问题在二维空间线性不可分。

可以将样本从原始空间映射到一个更高维得特征空间，使得样本在高维空间线性可分。

如果原始空间是有限维，即属性数有限，那么一定存在一个高维特征空间使样本可分。

令 $\phi(x)$ 表示将x映射后得特征向量，则超平面可以表示为：
$$
f(x) = \omega^T \phi(x) + b
$$
则对偶问题可以转换为：
$$
\max_{\alpha} \lbrace\sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jy_iy_j\phi(x_i)^T\phi(x_j) \rbrace
$$
因为 $\phi(x_i)^T\phi(x_j)$ 难以计算，可以设想一个函数：
$$
\kappa(x_i, x_j) = <\phi(x_i), \phi(x_j)> = \phi(x_i)^T \phi(x_j)
$$
即 $x_i, x_j$ 在特征空间得内积等于它们在原始样本空间中通过函数 $\kappa(\cdot, \cdot)$ 计算的结果。

### 支持向量回归

支持向量回归（support vector regression）假设我们能容忍 $f(x)$ 与 $y$ 之间最多有 $\epsilon$ 的偏差，即只有当 $f(x)$ 与 $y$ 之间的差别绝对值大于 $\epsilon$ 时才会计算损失。相当于以 $f(x)$ 为中心，构建了一个宽度为 $2\epsilon$ 的间隔带，若训练样本落入此间隔带，则被认为是预测正确的。

SVR形式为：
$$
\min_{\omega, b} \frac{1}{2}{||\omega||^2} + C\sum_{i=1}^{n}l_{\epsilon}(f(x_i) - y_i) \\
l_\epsilon(z) = \begin{cases}
				0,  &\mbox{if} &|z| \leqslant \epsilon; \\
				|z|-\epsilon, &\mbox{otherwise}
				\end{cases}
$$

## ch-7贝叶斯分类器

### 贝叶斯决策论

在概率框架下实施决策

假设有 $N$ 种可能的类别标记，即 $y = \lbrace c_1, c_2,..., c_N \rbrace$ ， $\lambda_{ij}$ 是将一个真实标记为 $c_j$ 的样本误分类为 $c_i$  所产生的损失。则基于后验概率的期望损失为：
$$
R(c_i|x) = \sum_{j=1}^{N} \lambda_{ij} P(c_j|x)
$$
最小化分类错误率：
$$
\lambda_{ij}= \begin {cases}
				0, &{if} &i=j; \\
				1, &{otherwise}
				\end{cases}
$$
此时，条件风险为：
$$
R(c|x) = 1 - P(c|x)
$$
于是，最小化分类错误率的贝叶斯最优分类器为：
$$
h^*(x)=\arg_{c\in{Y}}\max{P(c|x)}
$$
后验概率 $P(c|x)$ 很重要

机器学习所要实现的是基于有限的训练样本集尽可能准确地估计后验概率

- 判别式模型：给定 $x$ ，直接建模 $P(c|x)$ 来预测c， 称为判别式模型
- 生成式模型：对联合概率分布 $P(c,x)$ 进行建模

 对生成式模型，考虑：
$$
\begin{aligned}
P(c|x) &= \frac{P(c, x)}{P(x)} \\
		&= \frac{P(c)P(x|c)}{P(x)}
\end{aligned}
$$
其中，$P(c)$ 是类先验概率，$P(x|c)$ 是样本 $x$ 相对于类标记地类条件概率，或称为似然。

### 极大似然估计

估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进项估计。

假定 $P(x|c)$ 具有确定的形式并且被参数向量 $\theta_c$ 唯一确定，则任务变为利用训练集 $D$ 估计参数 $\theta_c$ ，即将 $P(x|c)$ 变换为：$P(x|\theta_c)$

概率模型的训练过程就是参数估计过程，对于参数估计，

- 频率主义学派：参数虽然未知，但却是客观存在的固定值，可以通过优化似然函数确定参数值
- 贝叶斯学派：参数是未观察到的随机变量，其本身也可由分布，可假定参数服从一个先验分布，然后基于观测到的数据来计算参数的后验分布

我们这里采用频率主义学派的极大似然估计（Maximum Likelihood Estimation）

用 $D_c$ 表示D中第c类样本组成的集合，则参数 $\theta_c$ 对于 $D_c$ 的似然为：
$$
P(D_c|\theta_c) = \prod_{x\in{D_c}} P(x|\theta_c)
$$
通常使用对数似然：
$$
LL(\theta_c) = \sum_{x\in{D_c}} \log{P(x|\theta_c)}
$$

### 朴素贝叶斯分类器

类条件概率 $P(x|c)$ 是所有属性上的联合概率，难以从有限的训练样本中直接估计，因而采用属性条件独立性假设。
$$
P(c|x) = \frac{P(c)P(x|c)}{P(x)} = \frac{P(c)}{P(x)} \prod_{i=1}^{d}P(x_i|c)
$$
其中d为属性数目

为了防止出现某一属性没有出现而导致概率为0，可以进行平滑操作，常用的为拉普拉斯修正。

### 半朴素贝叶斯分类器

属性条件独立性假说往往在现实生活很难实现，因而对这个假设进行一定程度的放松，称为半朴素贝叶斯分类器。

适当考虑一部分属性间的相互依赖信息，从而既不需要进行完全联合概率计算，又不至于彻底忽略比较强的属性依赖关系。

### 贝叶斯网

贝叶斯网又叫信念网，借助有向无环图来刻画属性之间的依赖关系，并使用条件概率表CPT来描述属性的联合概率分布。

给定父结点集，贝叶斯网假设每个属性与它的非后裔属性独立

贝叶斯网中三个变量的依赖关系：同父结构，V型结构，顺序结构

V型结构存在边际独立性

分析有向图中的条件独立性，可以使用有向分离：

- 先将有向图转换为无向图：找出有向图中所有的V型结构，在V型结构两个父结点间加一条无向边。
- 然后将所有有向边变为无向边，由此产生的无向图为道德图，父结点相连的过程称为道德化。

当属性间的依赖关系已知，只需要对训练样本计数，估计出每个结点的条件概率表即可。

但网络结构往往并不知晓，可以通过评分搜索来找出结构恰当的网。

- 首先定义一个评分函数，来估计贝叶斯网和训练数据的契合程度
- 然后基于评分函数寻找结构最优的贝叶斯网

常用评分函数通常基于信息论准则，将学习问题看作是数据压缩任务，学习的目标是找到一个能以最短编码长度描述训练数据的模型，对于贝叶斯网学习，模型就是一个贝叶斯网。

### EM算法

现实生活中往往会遇到不完整的训练样本，存在变量的观测值未知，即存在未观测变量

未观测变量称为隐变量

对隐变量可以通过计算其期望，最大化已观测数据的对数边际似然

EM（Expectation Maximization）算法是常用的估计参数隐变量的算法，它是一种迭代式算法：

- 若参数 $\theta$ 已知，则可根据训练数据推断出最优隐变量 $Z$ 的值，反之，若 $Z$ 的值已知，则可以方便地对参数 $\theta$ 作极大似然估计

## ch8-集成学习

### 个体与集成

集成学习通过构建并结合多个学习器来完成学习任务，又称多分类器系统，基于委员会地学习

先产生一组个体学习器，再用某种策略将它们结合起来

个体学习器通常由一个现有的学习算法从训练数据产生，如C4.5决策树等

集成分为同质集成与异质集成。

同质集成中的个体学习器称为基学习器，异质集成中的个体学习器称为组件学习器 component learner

集成学习很多理论研究针对弱学习器，而基学习器有时被直接称为弱学习器

一个简单的分析：

对于二分类问题 $y \in {\lbrace -1, +1 \rbrace}$ 以及真是函数 $f$ ，假定基学习器的错误率为 $\epsilon$ ，则对于每个基学习器 $h_i$ 有：
$$
P(h_i(x) \neq f(x)) = \epsilon
$$
将T个学习器进行结合：
$$
F(x) = sign(\sum_{i=1}^{T}h_i(x))
$$
由hoeffding不等式：
$$
P(\frac{\sum_{i=1}^{m}x_i}{m} - \frac{\sum_{i=1}^{m}E(x_i)}{m} \leqslant -\epsilon) \leqslant \exp(-2m\epsilon^2)
$$
进行推导：
$$
P(F(x) \neq f(x)) \leqslant \exp(-\frac{1}{2}T(1-2\epsilon)^2)
$$
当集成中个体分类器数目T增大时，集成的错误率将指数级下降

根据个体学习器的生成方式，集成学习可以分成两大类：

- 个体学习器存在强依赖关系，必须串行生成的序列化方法，代表为Boosting
- 个体学习器不存在强依赖关系，可同时生成的并行化方法，代表为Bagging和随机森林

### Boosting

算法的工作机制：先从初试训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布训练下一个基学习器，直至基学习器的数量达到T。

最著名的算法为：AdaBoost，基于基学习器的线性组合
$$
H(x) = \sum_{t=1}^{T} \alpha_th_t(x) \\
l_{exp}(H|D) = E[e^{-f(x)H(x)}]
$$

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # 训练弱分类器
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=weights)

            # 预测结果
            predictions = estimator.predict(X)

            # 计算误差率
            error = weights.dot(predictions != y)

            # 计算弱分类器权重
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.estimators.append(estimator)
            self.alphas.append(alpha)

            # 更新样本权重
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for estimator, alpha in zip(self.estimators, self.alphas):
            predictions += alpha * estimator.predict(X)
        return np.sign(predictions)

```

### Bagging与随机森林

给定一个数据集，一种可能的做法是对训练样本进行采样，产生出若干个不同的子集，再从每个数据子集中训练出一个基学习器。由于训练数据的不同，我们获得的基学习器可望具有比较大的差异。但是差异不能太大，可以使用相互有交叠的采样子集。

#### Bagging

直接基于自主采样法。给定包含m个样本的数据集，先随机取出一个样本放入采样集，再把该样本放回原始数据集。经过m次采样，得到大小为m的采样集。

采样出T个大小为m的采样集，基于每个采样集合训练出基学习器，再将这些学习器进行组合。

#### Random Forest

### 结合策略

学习器结合有三个方面的好处：

- 统计学角度看，由于学习任务的假设空间往往很大，可能有多个假设在训练集上达到同等性能，此时若使用单学习器可能因误选而导致泛化性能不佳，结合多个学习器则会减少这一风险。
- 计算角度看，学习算法往往陷入局部极小，有的局部极小泛化性能很差，通过多次运行后进行结合，可降低陷入糟糕局部极小点的风险。
- 表示的角度看，结合多个学习器，相应的假设空间有所扩大，有可能学得更好的近似。

常见的结合策略：

- 简单平均法

- 加权平均法——个体学习器相差较大
- 投票法：绝对多数，相对多数，加权
- 学习法：将个体学习器称为初级学习器，用于结合的学习器称为次级学习器或原学习器

经典学习法：stacking

- 先从初始训练集训练出初级学习器，然后生成一个新数据集用于训练次级学习器

### 多样性

误差-分歧分解

多样性度量：度量集成中个体分类器的多样性，即估算个体学习器的多样化程度

- 不合度量
- 相关系数
- Q-统计量
- $\kappa$ 统计量

## ch9-聚类

### 聚类任务

无监督学习中，训练样本的标记信息是未知的

聚类试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个簇（cluster），比如西瓜分为浅色，深色，本地，外地，需要注意的是，这些概念对聚类算法事先是未知的。

样本集 $D=\lbrace x_1, x_2,...,x_m\rbrace$ 包含m个无标记样本，每个样本是一个n维特征向量，聚类算法将样本集划分为k类，每个类之间无交集，且其并集是D，使用 $\lambda_j \in \lbrace1,2,3,...,k\rbrace$ 表示样本的样本 $x_j$ 的簇标记

则聚类的返回结果是一个包含m个元素的簇标记向量 $\lambda = \lbrace\lambda_1; \lambda_2;...;\lambda_m\rbrace$ 

聚类有两个基本问题：性能度量，距离计算

### 性能度量

性能度量又称有效性指标，通过性能度量评估聚类结果的好坏，另一方面，如果确定最终使用的性能度量，则可将其作为聚类过程的优化目标

通常，需要确保聚类结果的簇内相似度高且簇间相似度低

性能度量可以分为两种：一类是将聚类结果与某个参考模型进行比较，称为外部指标。另一类是直接考察聚类结果，称为内部指标

外部指标

聚类算法的类为 $C =\lbrace C_1,...,C_k\rbrace$ 参考模型为 $C^*=\lbrace C^*_1,...,C^*_s \rbrace$ 
$$
a = |SS|,b=|SD|,c=|DS|,d=|DD|
$$


- Jaccard系数 $JC=\frac{a}{a+b+c}$
- FM指数 $FMI=\sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}$
- Rand指标 $RI=\frac{2(a+d)}{m(m-1)}$

内部指标

- DB指数
- Dunn指数

### 距离度量

闵可夫斯基距离

曼哈顿距离

欧式距离

闵可夫斯基距离可用于有序属性，而对于无序属性，则需要采用VDM距离
$$
VDM_p(a, b)=\sum_{i=1}^{k} |\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}|^p
$$
其中，$m_{u,a,i}$ 表示第i个簇中，属性u取值为a的样本数，上式计算属性u上两个离散值a与b之间的距离

### 原型聚类

假设聚类结构可以通过一组原型刻画

#### k均值算法（k-means）

损失函数刻画为每一个类中，所有样本距离各个类中心点的距离和

算法采用贪心策略，通过迭代优化来近似求解

#### 学习向量量化 learning vector quantization

算法假设样本带有类别标记，利用样本的监督信息来辅助聚类

#### 高斯混合聚类

### 密度聚类

假设聚类结构能通过样本分布的紧密程度确定

DBSCAN算法：基于一组邻域参数来刻画样本分布的紧密程度

-  $\epsilon$ 邻域，对 $x_j \in D$ ，其邻域包含所有与其距离不大于 $\epsilon$ 的点，记为 $N_\epsilon(x_j)$ 
- 核心对象，若 $x_j$ 的 $\epsilon$ 邻域包含 $MinPts$ 个样本，则 $x_j$ 是一个核心对象
- 密度直达：若 $x_j$ 位于 $x_i$ 的邻域中，且 $x_i$ 是一个核心对象，则 $x_j$ 由 $x_i$ 密度直达
- 密度可达：类似图顶点的连通
- 密度相连：两个点都由第三个点密度可达

### 层次聚类

AGNES是一种采用自底向上聚合策略的层次聚类算法：先将数据集中的每个样本看作是一个初始聚类簇，然后在算法运行的每一步中找出距离最近的两个聚类簇进行合并，不断重复该过程，直到达到预设的聚类簇个数。

### 阅读材料

异常检测（anomaly detection）常借助聚类或距离计算进行，如将原理所有簇中心的样本作为异常点，或将密度极低处的样本作为异常点。也可以基于隔离性可快速检测出异常点。

## ch10-降维与度量学习

