import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm # 该_BatchNorm模块用于初始化批量归一化层

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger
""" from labml import experiment, lab, tracker, monit, logger
from labml_helpers.module import Module
from labml.utils.download import download_file # This module provides a simple way to download files from the internet.
from labml_nn.experiments.nlp_autoregression import transpose_batch # This module provides a simple way to define PyTorch modules.
from labml_nn.optimizers.noam import Noam # This module provides a simple way to define PyTorch modules.
from labml_nn.transformers import Encoder, MultiHeadAttention # This module provides a simple way to define PyTorch modules.
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.utils import subsequent_mask # This module provides a simple way to define PyTorch modules.
from labml_nn.transformers.models import EmbeddingsWithPositionalEncoding, TransformerLayer """
# 该脚本从 PyTorch、torchvision 和其他库导入各种模块和函数。这些模块对于构建和训练神经网络至关重要

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs): # 该函数用于初始化网络权重
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    # 采用 PyTorch 模块列表 ( module_list) 并对卷积层和线性层执行权重初始化
    if not isinstance(module_list, list): # 如果 module_list 不是列表，则将其转换为列表
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale # 该scale参数允许缩放初始化权重，特别是对于残差块
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill) # 该bias_fill参数用于填充偏差值
            elif isinstance(m, nn.Linear): # 该nn.Linear模块用于初始化线性层
                init.kaiming_normal_(m.weight, **kwargs) # 该kwargs参数可用于指定权重初始化的附加参数
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg): # 该函数用于堆叠相同的块以生成层
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential. # 返回一个nn.Sequential包含堆叠块的 PyTorch
    """
    layers = []
    for _ in range(num_basic_block): # 该for循环用于堆叠相同的块以生成层
        layers.append(basic_block(**kwarg)) # 该basic_block参数用于指定块的类型
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module): # 该ResidualBlockNoBN模块用于定义没有BN(批量归一化)的残差块
    # 该块通常用于残差网络
    """Residual block without BN.
# 它由两个卷积层和一个 ReLU 激活函数组成。
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """
    # 权重初始化可以使用 PyTorch 默认初始化或自定义初始化来执行。如果 pytorch_init 为 True，则使用 PyTorch 默认初始化。
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True) # 该nn.Conv2d模块用于定义卷积层
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True) # 该bias参数用于指定是否使用偏差
        self.relu = nn.ReLU(inplace=True) # 该nn.ReLU模块用于定义ReLU激活函数

        if not pytorch_init: # 该pytorch_init参数用于指定是否使用 PyTorch 默认初始化
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x): # 该forward函数用于定义前向传播
        identity = x # 该identity变量用于保存输入
        out = self.conv2(self.relu(self.conv1(x))) # 该out变量用于保存输出
        return identity + out * self.res_scale # 该res_scale参数用于指定残差比例

""" class AutoregressiveModel(nn.Module): # class for AutoregressiveModel
    """
    ## Auto regressive model
"""

    def __init__(self, src_embed: Module, encoder: Encoder, generator: Module): # constructor for AutoregressiveModel,src_embed,encoder,generator are the parameters, src_embed is the token embedding module, encoder is the transformer based encoder, generator is the next token generation layer
        super().__init__()  # super() function returns an object that represents the parent class.
        # Token embedding module
        self.src_embed = src_embed # token embedding module
        # Transformer based encoder
        self.encoder = encoder # transformer based encoder, this will be initialized on the first call,encoder is the transformer based encoder
        # Next token generation layer;
        # this gives logits of the the next token
        self.generator = generator # next token generation layer, this will be initialized on the first call, this give logits  of the the next token
        # This will be initialized on the first call
        self.src_mask = None # src_mask被初始化为None

    def forward(self, src: torch.Tensor): # 自回归的模型，src是张量
        # Create subsequent mask, so that the transformer can only pay attention to past tokens.
        if self.src_mask is None or self.src_mask.size(0) != len(src): # 如果src_mask是None或者src_mask的大小不等于src的长度
            self.src_mask = subsequent_mask(len(src)).to(src.device) # src_mask被初始化为subsequent_mask(len(src)).to(src.device)
        # Embed the tokens (`src`) and run it through the the transformer
        res = self.encoder(self.src_embed(src), self.src_mask) # 将tokens（`src`）嵌入并通过transformer运行
        # Generate logits of the next token
        return self.generator(res) # 生成下一个token的逻辑

class Configs: #配置类
    """
     ### Configurations
"""
    d_model: int = 512 # d_model是512，表示模型的维度
    n_layers: int = 6 # n_layers是6，表示层数
    n_heads: int = 8 # n_heads是8，表示头数
    dropout: float = 0.1 # dropout是0.1，表示丢弃概率
    d_ff: int = 2048 # d_ff是2048，表示前馈网络的维度
    glu_variant: str = 'GLU' # glu_variant是'GLU'，表示GLU变体
    epochs: int = 5 # epochs是5，表示训练的轮数
    grad_norm_clip: float = 0.5 # grad_norm_clip是0.5，表示梯度裁剪的范数

class Trainer: # Trainer类
    """
    ## Trainer
"""
    def __init__(self, configs: Configs): # 构造函数，configs是参数

        # FFN with Gated Linear Unit
        # $$FFN_{GLU}(x)(x, W_1, V, W_2) = (\sigma(x W_1) \otimes x V) W_2$$
        if configs.glu_variant == 'GLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.Sigmoid(), True, False, False, False) # FFN是GLU，使用FeedForward模块，d_model是模型的维度，d_ff是前馈网络的维度，dropout是丢弃概率，nn.Sigmoid()是激活函数，True表示使用bias，False表示使用LayerNorm，False表示使用Residual，False表示使用Skip Connection
        # FFN with Bilinear hidden layer
        # $$FFN_{Bilinear}(x)(x, W_1, V, W_2) = (x W_1 \otimes x V) W_2$$
        elif configs.glu_variant == 'Bilinear':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.Identity(), True, False, False, False) # nn.Identity()是激活函数
        # FFN with ReLU gate
        # $$FFN_{ReGLU}(x)(x, W_1, V, W_2) = (\max(0, x W_1) \otimes x V) W_2$$
        elif configs.glu_variant == 'ReGLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.ReLU(), True, False, False, False) # nn.ReLU()是激活函数
        # FFN with GELU gate
        # $$FFN_{GEGLU}(x)(x, W_1, V, W_2) = (\text{GELU}(x W_1) \otimes x V) W_2$$
        elif configs.glu_variant == 'GEGLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.GELU(), True, False, False, False) # nn.GELU()是激活函数
        # FFN with Swish gate
        # $$FFN_{SwiGLU}(x)(x, W_1, V, W_2) = (\text{Swish}_1(x W_1) \otimes x V) W_2$$
        # where $\text{Swish}_\beta(x) = x \sigma(\beta x)$
        elif configs.glu_variant == 'SwiGLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.SiLU(), True, False, False, False) # nn.SiLU()是激活函数
        # FFN with ReLU activation
        # $$FFN_{ReLU}(x)(x, W_1, W_2, b_1, b_2) = \text{ReLU}_1(x W_1 + b_1) W_2 + b_2$$
        elif configs.glu_variant == 'ReLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.ReLU()) # nn.ReLU()是激活函数
        # FFN with ReLU activation
        # $$FFN_{GELU}(x)(x, W_1, W_2, b_1, b_2) = \text{GELU}_1(x W_1 + b_1) W_2 + b_2$$
        elif configs.glu_variant == 'GELU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.GELU()) # nn.GELU()是激活函数
        else:
            raise ValueError(f'Unknown variant {configs.glu_variant}') # 如果不是上述的变体，则抛出异常



        # Initialize [Multi-Head Attention module](../mha.html)
        mha = MultiHeadAttention(configs.n_heads, configs.d_model, configs.dropout) # 初始化多头注意力模块
        # Initialize the [Transformer Block](../models.html#TransformerLayer)
        transformer_layer = TransformerLayer(d_model=configs.d_model, self_attn=mha, src_attn=None,
                                             feed_forward=ffn, dropout_prob=configs.dropout) # 初始化transformer层, d_model是模型的维度，self_attn是自注意力模块，feed_forward是前馈网络模块，dropout_prob是丢弃概率
        # Initialize the model with an
        # [embedding layer](../models.html#EmbeddingsWithPositionalEncoding)
        # (with fixed positional encoding)
        # [transformer encoder](../models.html#Encoder) and
        # a linear layer to generate logits.
        self.model = AutoregressiveModel(EmbeddingsWithPositionalEncoding(configs.d_model),
                                         Encoder(transformer_layer, configs.n_layers),
                                         nn.Linear(configs.d_model)) # 初始化模型，包括嵌入层，transformer编码器，线性层, d_model是模型的维度，n_chars是不同字符的数量

        # Move the model to the current device
        self.model.to(self.device) # 将模型移动到当前设备

        # Initialize [Noam optimizer](../../optimizers/noam.html)
        self.optimizer = Noam(self.model.parameters(), lr=1.0, warmup=2_000, d_model=configs.d_model) # 初始化Noam优化器，学习率是1.0，warmup是2000，d_model是模型的维度

        # Cross-entropy loss
        self.loss_func = nn.CrossEntropyLoss() # 交叉熵损失 """

class Upsample(nn.Sequential): # 该Upsample模块用于定义上采样模块
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3. # 该scale参数用于指定上采样比例
        num_feat (int): Channel number of intermediate features. # 该num_feat参数用于指定中间特征的通道数
    """

    def __init__(self, scale, num_feat): # 该__init__函数用于初始化上采样模块
        m = [] # 该m变量用于保存上采样模块
        if (scale & (scale - 1)) == 0:  # scale = 2^n # 它使用按位运算检查是否scale是 2 的幂。如果是，代码将进入循环来定义上采样层
            for _ in range(int(math.log(scale, 2))): #运行scale指定的以2为底的对数次数。在循环内，它将两层附加到m列表中
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1)) # 2D卷积层，将通道数增加4倍，并使用3x3内核，padding和stride 为1
                m.append(nn.PixelShuffle(2)) # 执行2倍像素混洗的层，可有效将空间分辨率提高2倍
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1)) # 该nn.Conv2d模块用于定义卷积层
            m.append(nn.PixelShuffle(3))
        else: # 如果scale不是2的幂或3，则引发错误
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m) # 该super函数用于调用父类的__init__函数


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
