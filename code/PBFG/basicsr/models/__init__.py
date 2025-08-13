import importlib #
from copy import deepcopy # from copy import deep
from os import path as osp # from os import path as osp

from basicsr.utils import get_root_logger, scandir #
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model'] # 它定义了一个名为__all__的列表，指定使用该from <module> import *语句时将导出哪些符号。在本例中，它导出该build_model函数

# automatically scan and import model modules for registry 自动查找和导入模型模块。
# scan all the files under the 'models' folder and collect files ending with '_model.py' 扫描'models'文件夹中以'_model.py'结尾的Python文件。
model_folder = osp.dirname(osp.abspath(__file__)) #build_model函数接受一个opt参数，该参数是一个配置字典。它使用配置中的model_type键来使用MODEL_REGISTRY创建指定模型类型的实例
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'basicsr.models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
