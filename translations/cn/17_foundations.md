# 第十七章：基础神经网络



本章开始了一段旅程，我们将深入研究我们在前几章中使用的模型的内部。我们将涵盖许多我们以前见过的相同内容，但这一次我们将更加密切地关注实现细节，而不那么密切地关注事物为什么是这样的实际问题。

我们将从头开始构建一切，仅使用对张量的基本索引。我们将从头开始编写一个神经网络，然后手动实现反向传播，以便我们在调用`loss.backward`时确切地知道 PyTorch 中发生了什么。我们还将看到如何使用自定义*autograd*函数扩展 PyTorch，允许我们指定自己的前向和后向计算。

# 从头开始构建神经网络层

让我们首先刷新一下我们对基本神经网络中如何使用矩阵乘法的理解。由于我们正在从头开始构建一切，所以最初我们将仅使用纯 Python（除了对 PyTorch 张量的索引），然后在看到如何创建后，将纯 Python 替换为 PyTorch 功能。

## 建模神经元

神经元接收一定数量的输入，并为每个输入设置内部权重。它对这些加权输入求和以产生输出，并添加内部偏置。在数学上，这可以写成

<math alttext="o u t equals sigma-summation Underscript i equals 1 Overscript n Endscripts x Subscript i Baseline w Subscript i Baseline plus b" display="block"><mrow><mi>o</mi> <mi>u</mi> <mi>t</mi> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msub><mi>x</mi> <mi>i</mi></msub> <msub><mi>w</mi> <mi>i</mi></msub> <mo>+</mo> <mi>b</mi></mrow></math>

如果我们将输入命名为<math alttext="left-parenthesis x 1 comma ellipsis comma x Subscript n Baseline right-parenthesis"><mrow><mo>(</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>⋯</mo> <mo>,</mo> <msub><mi>x</mi> <mi>n</mi></msub> <mo>)</mo></mrow></math>，我们的权重为<math alttext="left-parenthesis w 1 comma ellipsis comma w Subscript n Baseline right-parenthesis"><mrow><mo>(</mo> <msub><mi>w</mi> <mn>1</mn></msub> <mo>,</mo> <mo>⋯</mo> <mo>,</mo> <msub><mi>w</mi> <mi>n</mi></msub> <mo>)</mo></mrow></math>，以及我们的偏置<math alttext="b"><mi>b</mi></math>。在代码中，这被翻译为以下内容：

```py
output = sum([x*w for x,w in zip(inputs,weights)]) + bias
```

然后将此输出馈送到非线性函数中，称为*激活函数*，然后发送到另一个神经元。在深度学习中，最常见的是*修正线性单元*，或*ReLU*，正如我们所见，这是一种花哨的说法：

```py
def relu(x): return x if x >= 0 else 0
```

然后通过在连续的层中堆叠许多这些神经元来构建深度学习模型。我们创建一个具有一定数量的神经元（称为*隐藏大小*）的第一层，并将所有输入链接到每个神经元。这样的一层通常称为*全连接层*或*密集层*（用于密集连接），或*线性层*。

它要求您计算每个`input`和具有给定`weight`的每个神经元的点积：

```py
sum([x*w for x,w in zip(input,weight)])
```

如果您对线性代数有一点了解，您可能会记得当您进行*矩阵乘法*时会发生许多这些点积。更准确地说，如果我们的输入在大小为`batch_size`乘以`n_inputs`的矩阵`x`中，并且如果我们已将神经元的权重分组在大小为`n_neurons`乘以`n_inputs`的矩阵`w`中（每个神经元必须具有与其输入相同数量的权重），以及将所有偏置分组在大小为`n_neurons`的向量`b`中，则此全连接层的输出为

```py
y = x @ w.t() + b
```

其中`@`表示矩阵乘积，`w.t()`是`w`的转置矩阵。然后输出`y`的大小为`batch_size`乘以`n_neurons`，在位置`(i,j)`上我们有这个（对于数学爱好者）：

<math alttext="y Subscript i comma j Baseline equals sigma-summation Underscript k equals 1 Overscript n Endscripts x Subscript i comma k Baseline w Subscript k comma j Baseline plus b Subscript j" display="block"><mrow><msub><mi>y</mi> <mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msub><mi>x</mi> <mrow><mi>i</mi><mo>,</mo><mi>k</mi></mrow></msub> <msub><mi>w</mi> <mrow><mi>k</mi><mo>,</mo><mi>j</mi></mrow></msub> <mo>+</mo> <msub><mi>b</mi> <mi>j</mi></msub></mrow></math>

或者在代码中：

```py
y[i,j] = sum([a * b for a,b in zip(x[i,:],w[j,:])]) + b[j]
```

转置是必要的，因为在矩阵乘积`m @ n`的数学定义中，系数`(i,j)`如下：

```py
sum([a * b for a,b in zip(m[i,:],n[:,j])])
```

所以我们需要的非常基本的操作是矩阵乘法，因为这是神经网络核心中隐藏的内容。

## 从头开始的矩阵乘法

让我们编写一个函数，计算两个张量的矩阵乘积，然后再允许我们使用 PyTorch 版本。我们只会在 PyTorch 张量中使用索引：

```py
import torch
from torch import tensor
```

我们需要三个嵌套的`for`循环：一个用于行索引，一个用于列索引，一个用于内部求和。`ac`和`ar`分别表示`a`的列数和行数（对于`b`也是相同的约定），我们通过检查`a`的列数是否与`b`的行数相同来确保计算矩阵乘积是可能的：

```py
def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): c[i,j] += a[i,k] * b[k,j]
    return c
```

为了测试这一点，我们假装（使用随机矩阵）我们正在处理一个包含 5 个 MNIST 图像的小批量，将它们展平为`28*28`向量，然后使用线性模型将它们转换为 10 个激活值：

```py
m1 = torch.randn(5,28*28)
m2 = torch.randn(784,10)
```

让我们计时我们的函数，使用 Jupyter 的“魔术”命令`%time`：

```py
%time t1=matmul(m1, m2)
```

```py
CPU times: user 1.15 s, sys: 4.09 ms, total: 1.15 s
Wall time: 1.15 s
```

看看这与 PyTorch 内置的`@`有什么区别？

```py
%timeit -n 20 t2=m1@m2
```

```py
14 µs ± 8.95 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

正如我们所看到的，在 Python 中三个嵌套循环是一个坏主意！Python 是一种慢速语言，这不会高效。我们在这里看到 PyTorch 比 Python 快大约 100,000 倍——而且这还是在我们开始使用 GPU 之前！

这种差异是从哪里来的？PyTorch 没有在 Python 中编写矩阵乘法，而是使用 C++来加快速度。通常，当我们在张量上进行计算时，我们需要*向量化*它们，以便利用 PyTorch 的速度，通常使用两种技术：逐元素算术和广播。

## 逐元素算术

所有基本运算符（`+`、`-`、`*`、`/`、`>`、`<`、`==`）都可以逐元素应用。这意味着如果我们为两个具有相同形状的张量`a`和`b`写`a+b`，我们将得到一个由`a`和`b`元素之和组成的张量：

```py
a = tensor([10., 6, -4])
b = tensor([2., 8, 7])
a + b
```

```py
tensor([12., 14.,  3.])
```

布尔运算符将返回一个布尔数组：

```py
a < b
```

```py
tensor([False,  True,  True])
```

如果我们想知道`a`的每个元素是否小于`b`中对应的元素，或者两个张量是否相等，我们需要将这些逐元素操作与`torch.all`结合起来：

```py
(a < b).all(), (a==b).all()
```

```py
(tensor(False), tensor(False))
```

像`all`、`sum`和`mean`这样的缩减操作返回只有一个元素的张量，称为*秩-0 张量*。如果要将其转换为普通的 Python 布尔值或数字，需要调用`.item`：

```py
(a + b).mean().item()
```

```py
9.666666984558105
```

逐元素操作适用于任何秩的张量，只要它们具有相同的形状：

```py
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
m*m
```

```py
tensor([[ 1.,  4.,  9.],
        [16., 25., 36.],
        [49., 64., 81.]])
```

但是，不能对形状不同的张量执行逐元素操作（除非它们是可广播的，如下一节所讨论的）：

```py
n = tensor([[1., 2, 3], [4,5,6]])
m*n
```

```py
 RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at
 dimension 0
```

通过逐元素算术，我们可以去掉我们的三个嵌套循环中的一个：我们可以在将`a`的第`i`行和`b`的第`j`列对应的张量相乘之前对它们进行求和，这将加快速度，因为内部循环现在将由 PyTorch 以 C 速度执行。

要访问一列或一行，我们可以简单地写`a[i,：]`或`b[:,j]`。`:`表示在该维度上取所有内容。我们可以限制这个并只取该维度的一个切片，通过传递一个范围，比如`1:5`，而不仅仅是`:`。在这种情况下，我们将取第 1 到第 4 列的元素（第二个数字是不包括在内的）。

一个简化是我们总是可以省略尾随冒号，因此`a[i,:]`可以缩写为`a[i]`。考虑到所有这些，我们可以编写我们矩阵乘法的新版本：

```py
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc): c[i,j] = (a[i] * b[:,j]).sum()
    return c
```

```py
%timeit -n 20 t3 = matmul(m1,m2)
```

```py
1.7 ms ± 88.1 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

我们已经快了约 700 倍，只是通过删除那个内部的`for`循环！这只是开始——通过广播，我们可以删除另一个循环并获得更重要的加速。

## 广播

正如我们在第四章中讨论的那样，*广播*是由[Numpy 库](https://oreil.ly/nlV7Q)引入的一个术语，用于描述在算术操作期间如何处理不同秩的张量。例如，显然无法将 3×3 矩阵与 4×5 矩阵相加，但如果我们想将一个标量（可以表示为 1×1 张量）与矩阵相加呢？或者大小为 3 的向量与 3×4 矩阵？在这两种情况下，我们可以找到一种方法来理解这个操作。

广播为编码规则提供了特定的规则，用于在尝试进行逐元素操作时确定形状是否兼容，以及如何扩展较小形状的张量以匹配较大形状的张量。如果您想要能够编写快速执行的代码，掌握这些规则是至关重要的。在本节中，我们将扩展我们之前对广播的处理，以了解这些规则。

### 使用标量进行广播

使用标量进行广播是最简单的广播类型。当我们有一个张量`a`和一个标量时，我们只需想象一个与`a`形状相同且填充有该标量的张量，并执行操作：

```py
a = tensor([10., 6, -4])
a > 0
```

```py
tensor([ True,  True, False])
```

我们如何能够进行这种比较？`0`被*广播*以具有与`a`相同的维度。请注意，这是在不在内存中创建一个充满零的张量的情况下完成的（这将是低效的）。

如果要通过减去均值（标量）来标准化数据集（矩阵）并除以标准差（另一个标量），这是很有用的：

```py
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
(m - 5) / 2.73
```

```py
tensor([[-1.4652, -1.0989, -0.7326],
        [-0.3663,  0.0000,  0.3663],
        [ 0.7326,  1.0989,  1.4652]])
```

如果矩阵的每行有不同的均值怎么办？在这种情况下，您需要将一个向量广播到一个矩阵。

### 将向量广播到矩阵

我们可以将一个向量广播到一个矩阵中：

```py
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
m.shape,c.shape
```

```py
(torch.Size([3, 3]), torch.Size([3]))
```

```py
m + c
```

```py
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])
```

这里，`c`的元素被扩展为使三行匹配，从而使操作成为可能。同样，PyTorch 实际上并没有在内存中创建三个`c`的副本。这是由幕后的`expand_as`方法完成的：

```py
c.expand_as(m)
```

```py
tensor([[10., 20., 30.],
        [10., 20., 30.],
        [10., 20., 30.]])
```

如果我们查看相应的张量，我们可以请求其`storage`属性（显示用于张量的内存实际内容）来检查是否存储了无用的数据：

```py
t = c.expand_as(m)
t.storage()
```

```py
 10.0
 20.0
 30.0
[torch.FloatStorage of size 3]
```

尽管张量在官方上有九个元素，但内存中只存储了三个标量。这是可能的，这要归功于给该维度一个 0 步幅的巧妙技巧。在该维度上（这意味着当 PyTorch 通过添加步幅查找下一行时，它不会移动）：

```py
t.stride(), t.shape
```

```py
((0, 1), torch.Size([3, 3]))
```

由于`m`的大小为 3×3，有两种广播的方式。在最后一个维度上进行广播的事实是一种约定，这是来自广播规则的规定，与我们对张量排序的方式无关。如果我们这样做，我们会得到相同的结果：

```py
c + m
```

```py
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])
```

实际上，只有通过`n`，我们才能将大小为`n`的向量广播到大小为`m`的矩阵中：

```py
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6]])
c+m
```

```py
tensor([[11., 22., 33.],
        [14., 25., 36.]])
```

这不起作用：

```py
c = tensor([10.,20])
m = tensor([[1., 2, 3], [4,5,6]])
c+m
```

```py
 RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at
 dimension 1
```

如果我们想在另一个维度上进行广播，我们必须改变向量的形状，使其成为一个 3×1 矩阵。这可以通过 PyTorch 中的`unsqueeze`方法来实现：

```py
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]])
c = c.unsqueeze(1)
m.shape,c.shape
```

```py
(torch.Size([3, 3]), torch.Size([3, 1]))
```

这次，`c`在列侧进行了扩展：

```py
c+m
```

```py
tensor([[11., 12., 13.],
        [24., 25., 26.],
        [37., 38., 39.]])
```

与以前一样，只有三个标量存储在内存中：

```py
t = c.expand_as(m)
t.storage()
```

```py
 10.0
 20.0
 30.0
[torch.FloatStorage of size 3]
```

扩展后的张量具有正确的形状，因为列维度的步幅为 0：

```py
t.stride(), t.shape
```

```py
((1, 0), torch.Size([3, 3]))
```

使用广播，如果需要添加维度，则默认情况下会在开头添加。在之前进行广播时，PyTorch 在幕后执行了`c.unsqueeze(0)`：

```py
c = tensor([10.,20,30])
c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape
```

```py
(torch.Size([3]), torch.Size([1, 3]), torch.Size([3, 1]))
```

`unsqueeze`命令可以被`None`索引替换：

```py
c.shape, c[None,:].shape,c[:,None].shape
```

```py
(torch.Size([3]), torch.Size([1, 3]), torch.Size([3, 1]))
```

您可以始终省略尾随冒号，`...`表示所有前面的维度：

```py
c[None].shape,c[...,None].shape
```

```py
(torch.Size([1, 3]), torch.Size([3, 1]))
```

有了这个，我们可以在我们的矩阵乘法函数中删除另一个`for`循环。现在，我们不再将`a[i]`乘以`b[:,j]`，而是使用广播将`a[i]`乘以整个矩阵`b`，然后对结果求和：

```py
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous
        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0)
    return c
```

```py
%timeit -n 20 t4 = matmul(m1,m2)
```

```py
357 µs ± 7.2 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

现在我们比第一次实现快了 3700 倍！在继续之前，让我们更详细地讨论一下广播规则。

### 广播规则

在操作两个张量时，PyTorch 会逐个元素地比较它们的形状。它从*尾部维度*开始，逆向工作，在遇到空维度时添加 1。当以下情况之一为真时，两个维度是*兼容*的：

+   它们是相等的。

+   其中之一是 1，此时该维度会被广播以使其与其他维度相同。

数组不需要具有相同数量的维度。例如，如果您有一个 256×256×3 的 RGB 值数组，并且想要按不同值缩放图像中的每种颜色，您可以将图像乘以一个具有三个值的一维数组。根据广播规则排列这些数组的尾部轴的大小表明它们是兼容的：

```py
Image  (3d tensor): 256 x 256 x 3
Scale  (1d tensor):  (1)   (1)  3
Result (3d tensor): 256 x 256 x 3
```

然而，一个大小为 256×256 的 2D 张量与我们的图像不兼容：

```py
Image  (3d tensor): 256 x 256 x   3
Scale  (1d tensor):  (1)  256 x 256
Error
```

在我们早期的例子中，使用了一个 3×3 矩阵和一个大小为 3 的向量，广播是在行上完成的：

```py
Matrix (2d tensor):   3 x 3
Vector (1d tensor): (1)   3
Result (2d tensor):   3 x 3
```

作为练习，尝试确定何时需要添加维度（以及在何处），以便将大小为`64 x 3 x 256 x 256`的图像批次与三个元素的向量（一个用于均值，一个用于标准差）进行归一化。

另一种简化张量操作的有用方法是使用爱因斯坦求和约定。

## 爱因斯坦求和

在使用 PyTorch 操作`@`或`torch.matmul`之前，我们可以实现矩阵乘法的最后一种方法：*爱因斯坦求和*（`einsum`）。这是一种将乘积和求和以一般方式组合的紧凑表示。我们可以写出这样的方程：

```py
ik,kj -> ij
```

左侧表示操作数的维度，用逗号分隔。这里我们有两个分别具有两个维度（`i,k`和`k,j`）的张量。右侧表示结果维度，所以这里我们有一个具有两个维度`i,j`的张量。

爱因斯坦求和符号的规则如下：

1.  重复的索引会被隐式求和。

1.  每个索引在任何项中最多只能出现两次。

1.  每个项必须包含相同的非重复索引。

因此，在我们的例子中，由于`k`是重复的，我们对该索引求和。最终，该公式表示当我们在（`i,j`）中放入所有第一个张量中的系数（`i,k`）与第二个张量中的系数（`k,j`）相乘的总和时得到的矩阵……这就是矩阵乘积！

以下是我们如何在 PyTorch 中编写这段代码：

```py
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
```

爱因斯坦求和是一种非常实用的表达涉及索引和乘积和的操作的方式。请注意，您可以在左侧只有一个成员。例如，

```py
torch.einsum('ij->ji', a)
```

返回矩阵`a`的转置。您也可以有三个或更多成员：

```py
torch.einsum('bi,ij,bj->b', a, b, c)
```

这将返回一个大小为`b`的向量，其中第`k`个坐标是`a[k,i] b[i,j] c[k,j]`的总和。当您有更多维度时，这种表示特别方便，因为有批次。例如，如果您有两批次的矩阵并且想要计算每批次的矩阵乘积，您可以这样做：

```py
torch.einsum('bik,bkj->bij', a, b)
```

让我们回到使用`einsum`实现的新`matmul`，看看它的速度：

```py
%timeit -n 20 t5 = matmul(m1,m2)
```

```py
68.7 µs ± 4.06 µs per loop (mean ± std. dev. of 7 runs, 20 loops each)
```

正如您所看到的，它不仅实用，而且*非常*快。`einsum`通常是在 PyTorch 中执行自定义操作的最快方式，而无需深入研究 C++和 CUDA。（但通常不如精心优化的 CUDA 代码快，正如您从“从头开始的矩阵乘法”的结果中看到的。）

现在我们知道如何从头开始实现矩阵乘法，我们准备构建我们的神经网络——具体来说，是它的前向和后向传递——只使用矩阵乘法。

# 前向和后向传递

正如我们在第四章中看到的，为了训练一个模型，我们需要计算给定损失对其参数的所有梯度，这被称为*反向传播*。在*前向传播*中，我们根据矩阵乘积计算给定输入上模型的输出。当我们定义我们的第一个神经网络时，我们还将深入研究适当初始化权重的问题，这对于使训练正确开始至关重要。

## 定义和初始化一个层

首先我们将以两层神经网络为例。正如我们所看到的，一层可以表示为`y = x @ w + b`，其中`x`是我们的输入，`y`是我们的输出，`w`是该层的权重（如果我们不像之前那样转置，则大小为输入数量乘以神经元数量），`b`是偏置向量：

```py
def lin(x, w, b): return x @ w + b
```

我们可以将第二层叠加在第一层上，但由于数学上两个线性操作的组合是另一个线性操作，只有在中间放入一些非线性的东西才有意义，称为激活函数。正如本章开头提到的，在深度学习应用中，最常用的激活函数是 ReLU，它返回`x`和`0`的最大值。

在本章中，我们实际上不会训练我们的模型，因此我们将为我们的输入和目标使用随机张量。假设我们的输入是大小为 100 的 200 个向量，我们将它们分组成一个批次，我们的目标是 200 个随机浮点数：

```py
x = torch.randn(200, 100)
y = torch.randn(200)
```

对于我们的两层模型，我们将需要两个权重矩阵和两个偏置向量。假设我们的隐藏大小为 50，输出大小为 1（对于我们的输入之一，相应的输出在这个玩具示例中是一个浮点数）。我们随机初始化权重，偏置为零：

```py
w1 = torch.randn(100,50)
b1 = torch.zeros(50)
w2 = torch.randn(50,1)
b2 = torch.zeros(1)
```

然后我们的第一层的结果就是这样的：

```py
l1 = lin(x, w1, b1)
l1.shape
```

```py
torch.Size([200, 50])
```

请注意，这个公式适用于我们的输入批次，并返回一个隐藏状态批次：`l1`是一个大小为 200（我们的批次大小）乘以 50（我们的隐藏大小）的矩阵。

然而，我们的模型初始化方式存在问题。要理解这一点，我们需要查看`l1`的均值和标准差（std）：

```py
l1.mean(), l1.std()
```

```py
(tensor(0.0019), tensor(10.1058))
```

均值接近零，这是可以理解的，因为我们的输入和权重矩阵的均值都接近零。但标准差，表示我们的激活离均值有多远，从 1 变为 10。这是一个真正的大问题，因为这只是一个层。现代神经网络可以有数百层，因此如果每一层将我们的激活的规模乘以 10，到了最后一层，我们将无法用计算机表示数字。

实际上，如果我们在`x`和大小为 100×100 的随机矩阵之间进行 50 次乘法运算，我们将得到这个结果：

```py
x = torch.randn(200, 100)
for i in range(50): x = x @ torch.randn(100,100)
x[0:5,0:5]
```

```py
tensor([[nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan]])
```

结果是到处都是`nan`。也许我们的矩阵的规模太大了，我们需要更小的权重？但如果我们使用太小的权重，我们将遇到相反的问题-我们的激活的规模将从 1 变为 0.1，在 100 层之后，我们将到处都是零：

```py
x = torch.randn(200, 100)
for i in range(50): x = x @ (torch.randn(100,100) * 0.01)
x[0:5,0:5]
```

```py
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
```

因此，我们必须精确地缩放我们的权重矩阵，以使我们的激活的标准差保持在 1。我们可以通过数学计算出要使用的确切值，正如 Xavier Glorot 和 Yoshua Bengio 在[“理解训练深度前馈神经网络的困难”](https://oreil.ly/9tiTC)中所示。给定层的正确比例是<math alttext="1 slash StartRoot n Subscript i n Baseline EndRoot"><mrow><mn>1</mn> <mo>/</mo> <msqrt><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></msqrt></mrow></math>，其中<math alttext="n Subscript i n"><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></math>代表输入的数量。

在我们的情况下，如果有 100 个输入，我们应该将我们的权重矩阵缩放为 0.1：

```py
x = torch.randn(200, 100)
for i in range(50): x = x @ (torch.randn(100,100) * 0.1)
x[0:5,0:5]
```

```py
tensor([[ 0.7554,  0.6167, -0.1757, -1.5662,  0.5644],
        [-0.1987,  0.6292,  0.3283, -1.1538,  0.5416],
        [ 0.6106,  0.2556, -0.0618, -0.9463,  0.4445],
        [ 0.4484,  0.7144,  0.1164, -0.8626,  0.4413],
        [ 0.3463,  0.5930,  0.3375, -0.9486,  0.5643]])
```

终于，一些既不是零也不是`nan`的数字！请注意，即使经过了那 50 个虚假层，我们的激活的规模仍然是稳定的：

```py
x.std()
```

```py
tensor(0.7042)
```

如果你稍微调整一下 scale 的值，你会注意到即使从 0.1 稍微偏离，你会得到非常小或非常大的数字，因此正确初始化权重非常重要。

让我们回到我们的神经网络。由于我们稍微改变了我们的输入，我们需要重新定义它们：

```py
x = torch.randn(200, 100)
y = torch.randn(200)
```

对于我们的权重，我们将使用正确的 scale，这被称为*Xavier 初始化*（或*Glorot 初始化*）：

```py
from math import sqrt
w1 = torch.randn(100,50) / sqrt(100)
b1 = torch.zeros(50)
w2 = torch.randn(50,1) / sqrt(50)
b2 = torch.zeros(1)
```

现在如果我们计算第一层的结果，我们可以检查均值和标准差是否受控制：

```py
l1 = lin(x, w1, b1)
l1.mean(),l1.std()
```

```py
(tensor(-0.0050), tensor(1.0000))
```

非常好。现在我们需要经过一个 ReLU，所以让我们定义一个。ReLU 去除负数并用零替换它们，这另一种说法是它将我们的张量夹在零处：

```py
def relu(x): return x.clamp_min(0.)
```

我们通过这个激活：

```py
l2 = relu(l1)
l2.mean(),l2.std()
```

```py
(tensor(0.3961), tensor(0.5783))
```

现在我们回到原点：我们的激活均值变为 0.4（这是可以理解的，因为我们去除了负数），标准差下降到 0.58。所以像以前一样，经过几层后我们可能最终会得到零：

```py
x = torch.randn(200, 100)
for i in range(50): x = relu(x @ (torch.randn(100,100) * 0.1))
x[0:5,0:5]
```

```py
tensor([[0.0000e+00, 1.9689e-08, 4.2820e-08, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.6701e-08, 4.3501e-08, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.0976e-08, 3.0411e-08, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.8457e-08, 4.9469e-08, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 1.9949e-08, 4.1643e-08, 0.0000e+00, 0.0000e+00]])
```

这意味着我们的初始化不正确。为什么？在 Glorot 和 Bengio 撰写他们的文章时，神经网络中最流行的激活函数是双曲正切（tanh，他们使用的那个），而该初始化并没有考虑到我们的 ReLU。幸运的是，有人已经为我们计算出了正确的 scale 供我们使用。在[“深入研究整流器：超越人类水平的性能”](https://oreil.ly/-_quA)（我们之前见过的文章，介绍了 ResNet），Kaiming He 等人表明我们应该使用以下 scale 代替：<math alttext="StartRoot 2 slash n Subscript i n Baseline EndRoot"><msqrt><mrow><mn>2</mn> <mo>/</mo> <msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></mrow></msqrt></math>，其中<math alttext="n Subscript i n"><msub><mi>n</mi> <mrow><mi>i</mi><mi>n</mi></mrow></msub></math>是我们模型的输入数量。让我们看看这给我们带来了什么：

```py
x = torch.randn(200, 100)
for i in range(50): x = relu(x @ (torch.randn(100,100) * sqrt(2/100)))
x[0:5,0:5]
```

```py
tensor([[0.2871, 0.0000, 0.0000, 0.0000, 0.0026],
        [0.4546, 0.0000, 0.0000, 0.0000, 0.0015],
        [0.6178, 0.0000, 0.0000, 0.0180, 0.0079],
        [0.3333, 0.0000, 0.0000, 0.0545, 0.0000],
        [0.1940, 0.0000, 0.0000, 0.0000, 0.0096]])
```

好了：这次我们的数字不全为零。所以让我们回到我们神经网络的定义，并使用这个初始化（被称为*Kaiming 初始化*或*He 初始化*）：

```py
x = torch.randn(200, 100)
y = torch.randn(200)
```

```py
w1 = torch.randn(100,50) * sqrt(2 / 100)
b1 = torch.zeros(50)
w2 = torch.randn(50,1) * sqrt(2 / 50)
b2 = torch.zeros(1)
```

让我们看看通过第一个线性层和 ReLU 后激活的规模：

```py
l1 = lin(x, w1, b1)
l2 = relu(l1)
l2.mean(), l2.std()
```

```py
(tensor(0.5661), tensor(0.8339))
```

好多了！现在我们的权重已经正确初始化，我们可以定义我们的整个模型：

```py
def model(x):
    l1 = lin(x, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3
```

这是前向传播。现在剩下的就是将我们的输出与我们拥有的标签（在这个例子中是随机数）进行比较，使用损失函数。在这种情况下，我们将使用均方误差。（这是一个玩具问题，这是下一步计算梯度所使用的最简单的损失函数。）

唯一的微妙之处在于我们的输出和目标形状并不完全相同——经过模型后，我们得到这样的输出：

```py
out = model(x)
out.shape
```

```py
torch.Size([200, 1])
```

为了去掉这个多余的 1 维，我们使用`squeeze`函数：

```py
def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()
```

现在我们准备计算我们的损失：

```py
loss = mse(out, y)
```

前向传播到此结束，现在让我们看一下梯度。

## 梯度和反向传播

我们已经看到 PyTorch 通过一个神奇的调用`loss.backward`计算出我们需要的所有梯度，但让我们探究一下背后发生了什么。

现在我们需要计算损失相对于模型中所有权重的梯度，即`w1`、`b1`、`w2`和`b2`中的所有浮点数。为此，我们需要一点数学，具体来说是*链式法则*。这是指导我们如何计算复合函数导数的微积分规则：

<math alttext="left-parenthesis g ring f right-parenthesis prime left-parenthesis x right-parenthesis equals g prime left-parenthesis f left-parenthesis x right-parenthesis right-parenthesis f prime left-parenthesis x right-parenthesis" display="block"><mrow><msup><mrow><mo>(</mo><mi>g</mi><mo>∘</mo><mi>f</mi><mo>)</mo></mrow> <mo>'</mo></msup> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <msup><mi>g</mi> <mo>'</mo></msup> <mrow><mo>(</mo> <mi>f</mi> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>)</mo></mrow> <msup><mi>f</mi> <mo>'</mo></msup> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow></mrow></math>

# Jeremy 说

我发现这种符号很难理解，所以我喜欢这样想：如果 `y = g(u)` 和 `u=f(x)`，那么 `dy/dx = dy/du * du/dx`。这两种符号意思相同，所以使用任何一种都可以。

我们的损失是不同函数的大组合：均方误差（实际上是均值和平方的组合），第二个线性层，一个 ReLU，和第一个线性层。例如，如果我们想要损失相对于 `b2` 的梯度，而我们的损失由以下定义：

```py
loss = mse(out,y) = mse(lin(l2, w2, b2), y)
```

链式法则告诉我们我们有这个：

<math alttext="StartFraction d l o s s Over d b 2 EndFraction equals StartFraction d l o s s Over d o u t EndFraction times StartFraction d o u t Over d b 2 EndFraction equals StartFraction d Over d o u t EndFraction m s e left-parenthesis o u t comma y right-parenthesis times StartFraction d Over d b 2 EndFraction l i n left-parenthesis l 2 comma w 2 comma b 2 right-parenthesis" display="block"><mrow><mfrac><mrow><mtext>d</mtext><mi>l</mi><mi>o</mi><mi>s</mi><mi>s</mi></mrow> <mrow><mtext>d</mtext><msub><mi>b</mi> <mn>2</mn></msub></mrow></mfrac> <mo>=</mo> <mfrac><mrow><mtext>d</mtext><mi>l</mi><mi>o</mi><mi>s</mi><mi>s</mi></mrow> <mrow><mtext>d</mtext><mi>o</mi><mi>u</mi><mi>t</mi></mrow></mfrac> <mo>×</mo> <mfrac><mrow><mtext>d</mtext><mi>o</mi><mi>u</mi><mi>t</mi></mrow> <mrow><mtext>d</mtext><msub><mi>b</mi> <mn>2</mn></msub></mrow></mfrac> <mo>=</mo> <mfrac><mtext>d</mtext> <mrow><mtext>d</mtext><mi>o</mi><mi>u</mi><mi>t</mi></mrow></mfrac> <mi>m</mi> <mi>s</mi> <mi>e</mi> <mrow><mo>(</mo> <mi>o</mi> <mi>u</mi> <mi>t</mi> <mo>,</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>×</mo> <mfrac><mtext>d</mtext> <mrow><mtext>d</mtext><msub><mi>b</mi> <mn>2</mn></msub></mrow></mfrac> <mi>l</mi> <mi>i</mi> <mi>n</mi> <mrow><mo>(</mo> <msub><mi>l</mi> <mn>2</mn></msub> <mo>,</mo> <msub><mi>w</mi> <mn>2</mn></msub> <mo>,</mo> <msub><mi>b</mi> <mn>2</mn></msub> <mo>)</mo></mrow></mrow></math>

要计算损失相对于 <math alttext="b 2"><msub><mi>b</mi> <mn>2</mn></msub></math> 的梯度，我们首先需要损失相对于我们的输出 <math alttext="o u t"><mrow><mi>o</mi> <mi>u</mi> <mi>t</mi></mrow></math> 的梯度。如果我们想要损失相对于 <math alttext="w 2"><msub><mi>w</mi> <mn>2</mn></msub></math> 的梯度也是一样的。然后，要得到损失相对于 <math alttext="b 1"><msub><mi>b</mi> <mn>1</mn></msub></math> 或 <math alttext="w 1"><msub><mi>w</mi> <mn>1</mn></msub></math> 的梯度，我们将需要损失相对于 <math alttext="l 1"><msub><mi>l</mi> <mn>1</mn></msub></math> 的梯度，这又需要损失相对于 <math alttext="l 2"><msub><mi>l</mi> <mn>2</mn></msub></math> 的梯度，这将需要损失相对于 <math alttext="o u t"><mrow><mi>o</mi> <mi>u</mi> <mi>t</mi></mrow></math> 的梯度。

因此，为了计算更新所需的所有梯度，我们需要从模型的输出开始，逐层向后工作，一层接一层地——这就是为什么这一步被称为*反向传播*。我们可以通过让我们实现的每个函数（`relu`、`mse`、`lin`）提供其反向步骤来自动化它：也就是说，如何从损失相对于输出的梯度推导出损失相对于输入的梯度。

在这里，我们将这些梯度填充到每个张量的属性中，有点像 PyTorch 在`.grad`中所做的那样。

首先是我们模型输出（损失函数的输入）相对于损失的梯度。我们撤消了`mse`中的`squeeze`，然后我们使用给出<math alttext="x squared"><msup><mi>x</mi> <mn>2</mn></msup></math>的导数的公式：<math alttext="2 x"><mrow><mn>2</mn> <mi>x</mi></mrow></math>。均值的导数只是 1/*n*，其中*n*是我们输入中的元素数：

```py
def mse_grad(inp, targ):
    # grad of loss with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]
```

对于 ReLU 和我们的线性层的梯度，我们使用相对于输出的损失的梯度（在`out.g`中）并应用链式法则来计算相对于输出的损失的梯度（在`inp.g`中）。链式法则告诉我们`inp.g = relu'(inp) * out.g`。`relu`的导数要么是 0（当输入为负数时），要么是 1（当输入为正数时），因此这给出了以下结果：

```py
def relu_grad(inp, out):
    # grad of relu with respect to input activations
    inp.g = (inp>0).float() * out.g
```

计算损失相对于线性层中的输入、权重和偏差的梯度的方案是相同的：

```py
def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = inp.t() @ out.g
    b.g = out.g.sum(0)
```

我们不会深入讨论定义它们的数学公式，因为对我们的目的来说它们不重要，但如果你对这个主题感兴趣，可以查看可汗学院出色的微积分课程。

一旦我们定义了这些函数，我们就可以使用它们来编写后向传递。由于每个梯度都会自动填充到正确的张量中，我们不需要将这些`_grad`函数的结果存储在任何地方——我们只需要按照前向传递的相反顺序执行它们，以确保在每个函数中`out.g`存在：

```py
def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # we don't actually need the loss in backward!
    loss = mse(out, targ)

    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
```

现在我们可以在`w1.g`、`b1.g`、`w2.g`和`b2.g`中访问我们模型参数的梯度。我们已经成功定义了我们的模型——现在让我们让它更像一个 PyTorch 模块。

## 重构模型

我们使用的三个函数有两个相关的函数：一个前向传递和一个后向传递。我们可以创建一个类将它们包装在一起，而不是分开编写它们。该类还可以存储后向传递的输入和输出。这样，我们只需要调用`backward`：

```py
class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out

    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g
```

`__call__`是 Python 中的一个魔术名称，它将使我们的类可调用。当我们键入`y = Relu()(x)`时，将执行这个操作。我们也可以对我们的线性层和 MSE 损失做同样的操作：

```py
class Lin():
    def __init__(self, w, b): self.w,self.b = w,b

    def __call__(self, inp):
        self.inp = inp
        self.out = inp@self.w + self.b
        return self.out

    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)
```

```py
class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out

    def backward(self):
        x = (self.inp.squeeze()-self.targ).unsqueeze(-1)
        self.inp.g = 2.*x/self.targ.shape[0]
```

然后我们可以把一切都放在一个模型中，我们用我们的张量`w1`、`b1`、`w2`和`b2`来初始化：

```py
class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()
```

这种重构和将事物注册为模型的层的好处是，前向和后向传递现在非常容易编写。如果我们想要实例化我们的模型，我们只需要写这个：

```py
model = Model(w1, b1, w2, b2)
```

然后前向传递可以这样执行：

```py
loss = model(x, y)
```

然后使用这个进行后向传递：

```py
model.backward()
```

## 转向 PyTorch

我们编写的`Lin`、`Mse`和`Relu`类有很多共同之处，所以我们可以让它们都继承自同一个基类：

```py
class LayerFunction():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):  raise Exception('not implemented')
    def bwd(self):      raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)
```

然后我们只需要在每个子类中实现`forward`和`bwd`：

```py
class Relu(LayerFunction):
    def forward(self, inp): return inp.clamp_min(0.)
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g
```

```py
class Lin(LayerFunction):
    def __init__(self, w, b): self.w,self.b = w,b

    def forward(self, inp): return inp@self.w + self.b

    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = out.g.sum(0)
```

```py
class Mse(LayerFunction):
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    def bwd(self, out, inp, targ):
        inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
```

我们模型的其余部分可以与以前相同。这越来越接近 PyTorch 的做法。我们需要区分的每个基本函数都被写成一个`torch.autograd.Function`对象，它有一个`forward`和一个`backward`方法。PyTorch 将跟踪我们进行的任何计算，以便能够正确运行反向传播，除非我们将张量的`requires_grad`属性设置为`False`。

编写其中一个（几乎）和编写我们原始类一样容易。不同之处在于我们选择保存什么并将其放入上下文变量中（以确保我们不保存不需要的任何内容），并在`backward`传递中返回梯度。很少需要编写自己的`Function`，但如果您需要某些奇特的东西或想要干扰常规函数的梯度，这里是如何编写的：

```py
from torch.autograd import Function

class MyRelu(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.clamp_min(0.)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        return grad_output * (i>0).float()
```

用于构建利用这些`Function`的更复杂模型的结构是`torch.nn.Module`。这是所有模型的基本结构，到目前为止您看到的所有神经网络都是从该类中继承的。它主要有助于注册所有可训练的参数，正如我们已经看到的可以在训练循环中使用的那样。

要实现一个`nn.Module`，你只需要做以下几步：

1.  确保在初始化时首先调用超类`__init__`。

1.  将模型的任何参数定义为具有`nn.Parameter`属性。

1.  定义一个`forward`函数，返回模型的输出。

这里是一个从头开始的线性层的例子：

```py
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * sqrt(2/n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x): return x @ self.weight.t() + self.bias
```

正如您所看到的，这个类会自动跟踪已定义的参数：

```py
lin = LinearLayer(10,2)
p1,p2 = lin.parameters()
p1.shape,p2.shape
```

```py
(torch.Size([2, 10]), torch.Size([2]))
```

正是由于`nn.Module`的这个特性，我们可以只说`opt.step`，并让优化器循环遍历参数并更新每个参数。

请注意，在 PyTorch 中，权重存储为一个`n_out x n_in`矩阵，这就是为什么在前向传递中我们有转置的原因。

通过使用 PyTorch 中的线性层（也使用 Kaiming 初始化），我们在本章中一直在构建的模型可以这样编写：

```py
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out))
        self.loss = mse

    def forward(self, x, targ): return self.loss(self.layers(x).squeeze(), targ)
```

fastai 提供了自己的`Module`变体，与`nn.Module`相同，但不需要您调用`super().__init__()`（它会自动为您执行）：

```py
class Model(Module):
    def __init__(self, n_in, nh, n_out):
        self.layers = nn.Sequential(
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out))
        self.loss = mse

    def forward(self, x, targ): return self.loss(self.layers(x).squeeze(), targ)
```

在第十九章中，我们将从这样一个模型开始，看看如何从头开始构建一个训练循环，并将其重构为我们在之前章节中使用的内容。

# 结论

在本章中，我们探讨了深度学习的基础，从矩阵乘法开始，然后实现了神经网络的前向和反向传递。然后我们重构了我们的代码，展示了 PyTorch 在底层的工作原理。

以下是一些需要记住的事项：

+   神经网络基本上是一堆矩阵乘法，中间夹杂着非线性。

+   Python 很慢，所以为了编写快速代码，我们必须对其进行向量化，并利用诸如逐元素算术和广播等技术。

+   如果从末尾开始向后匹配的维度相同（如果它们相同，或者其中一个是 1），则两个张量是可广播的。为了使张量可广播，我们可能需要使用`unsqueeze`或`None`索引添加大小为 1 的维度。

+   正确初始化神经网络对于开始训练至关重要。当我们有 ReLU 非线性时，应使用 Kaiming 初始化。

+   反向传递是应用链式法则多次计算，从我们模型的输出开始，逐层向后计算梯度。

+   在子类化`nn.Module`时（如果不使用 fastai 的`Module`），我们必须在我们的`__init__`方法中调用超类`__init__`方法，并且我们必须定义一个接受输入并返回所需结果的`forward`函数。

# 问卷

1.  编写 Python 代码来实现一个单个神经元。

1.  编写实现 ReLU 的 Python 代码。

1.  用矩阵乘法的术语编写一个密集层的 Python 代码。

1.  用纯 Python 编写一个密集层的 Python 代码（即使用列表推导和内置到 Python 中的功能）。

1.  一个层的“隐藏大小”是什么？

1.  在 PyTorch 中，`t`方法是做什么的？

1.  为什么在纯 Python 中编写矩阵乘法非常慢？

1.  在`matmul`中，为什么`ac==br`？

1.  在 Jupyter Notebook 中，如何测量执行单个单元格所需的时间？

1.  什么是逐元素算术？

1.  编写 PyTorch 代码来测试 `a` 的每个元素是否大于 `b` 的对应元素。

1.  什么是秩为 0 的张量？如何将其转换为普通的 Python 数据类型？

1.  这返回什么，为什么？

    ```py
    tensor([1,2]) + tensor([1])
    ```

1.  这返回什么，为什么？

    ```py
    tensor([1,2]) + tensor([1,2,3])
    ```

1.  逐元素算术如何帮助我们加速 `matmul`？

1.  广播规则是什么？

1.  `expand_as` 是什么？展示一个如何使用它来匹配广播结果的示例。

1.  `unsqueeze` 如何帮助我们解决某些广播问题？

1.  我们如何使用索引来执行与 `unsqueeze` 相同的操作？

1.  我们如何显示张量使用的内存的实际内容？

1.  将大小为 3 的向量添加到大小为 3×3 的矩阵时，向量的元素是添加到矩阵的每一行还是每一列？（确保通过在笔记本中运行此代码来检查您的答案。）

1.  广播和 `expand_as` 会导致内存使用增加吗？为什么或为什么不？

1.  使用爱因斯坦求和实现 `matmul`。

1.  在 `einsum` 的左侧重复索引字母代表什么？

1.  爱因斯坦求和符号的三条规则是什么？为什么？

1.  神经网络的前向传播和反向传播是什么？

1.  为什么我们需要在前向传播中存储一些计算出的中间层的激活？

1.  具有标准差远离 1 的激活的缺点是什么？

1.  权重初始化如何帮助避免这个问题？

1.  初始化权重的公式是什么，以便在普通线性层和 ReLU 后跟线性层中获得标准差为 1？

1.  为什么有时我们必须在损失函数中使用 `squeeze` 方法？

1.  `squeeze` 方法的参数是做什么的？为什么可能很重要包含这个参数，尽管 PyTorch 不需要它？

1.  链式法则是什么？展示本章中提出的两种形式中的任意一种方程。

1.  展示如何使用链式法则计算 `mse(lin(l2, w2, b2), y)` 的梯度。

1.  ReLU 的梯度是什么？用数学或代码展示它。（您不应该需要记住这个—尝试使用您对函数形状的知识来弄清楚它。）

1.  在反向传播中，我们需要以什么顺序调用 `*_grad` 函数？为什么？

1.  `__call__` 是什么？

1.  编写 `torch.autograd.Function` 时我们必须实现哪些方法？

1.  从头开始编写 `nn.Linear` 并测试其是否有效。

1.  `nn.Module` 和 fastai 的 `Module` 之间有什么区别？

## 进一步研究

1.  将 ReLU 实现为 `torch.autograd.Function` 并用它训练模型。

1.  如果您对数学感兴趣，请确定数学符号中线性层的梯度。将其映射到本章中的实现。

1.  了解 PyTorch 中的 `unfold` 方法，并结合矩阵乘法实现自己的二维卷积函数。然后训练一个使用它的 CNN。

1.  使用 NumPy 而不是 PyTorch 在本章中实现所有内容。
