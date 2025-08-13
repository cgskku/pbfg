# 用于在较大的深度学习项目或库的上下文中构建神经网络架构
import importlib
from copy import deepcopy # 深拷贝
from os import path as osp # 路径

from basicsr.utils import get_root_logger, scandir # 日志
from basicsr.utils.registry import ARCH_REGISTRY # 注册

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__)) # 获取当前文件的绝对路径
# archs它收集文件夹中以“_arch.py”结尾的Python 文件的名称并导入这些模块
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'basicsr.archs.{file_name}') for file_name in arch_filenames] # 导入模块


def build_network(opt): # 根据输入选项 ( opt) 创建并返回神经网络
    opt = deepcopy(opt)
    network_type = opt.pop('type') # type从字典中提取密钥以opt确定要构建的网络的类型
    # 该函数在ARCH_REGISTRY中查找指定的网络类型，并使用提供的选项创建网络实例
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.') # 它记录有关创建的网络的信息
    return net
