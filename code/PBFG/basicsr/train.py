import datetime # 时间计算
import logging # 日志
import math # 数学计算
import time # 时间
import torch # pytorch
from os import path as osp # 路径
# import wandb
torch.cuda.empty_cache()
from basicsr.data import build_dataloader, build_dataset # build_dataloader创建数据加载器。处理数据加载、预处理和批处理； build_dataset设置用于训练和验证的数据源
from basicsr.data.data_sampler import EnlargedSampler # 扩大采样,数据增强
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher # 预取数据加载器
from basicsr.models import build_model # 用于创建深度学习模型的函数。会读取模型配置并相应地实例化模型架构
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
# check_resume检查是否需要恢复训练，get_root_logger获取根日志器，get_time_str获取时间字符串
# init_tb_logger初始化tensorboard日志器，init_wandb_logger初始化wandb日志器
# make_exp_dirs创建实验目录，mkdir_and_rename创建目录并重命名，scandir扫描目录
def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        # init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

def create_train_val_dataloader(opt, logger): # 据提供的配置选项为训练和验证数据集创建数据加载器
    # create train and val dataloaders
    train_loader, val_loaders = None, [] # 初始化，训练和验证数据加载器
    for phase, dataset_opt in opt['datasets'].items(): # 遍历数据集
        if phase == 'train': # 训练集
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt) #从训练数据集中采样数据。该采样器通过将数据集划分为每个 GPU 的更小的块来帮助进行分布式训练。
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader( # 创建数据加载器
                train_set, # 数据集
                dataset_opt, # 数据集选项
                num_gpu=opt['num_gpu'], # GPU数量
                dist=opt['dist'], # 分布式训练
                sampler=train_sampler, # 采样器
                seed=opt['manual_seed']) # 随机种子
               
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter']) # 总迭代次数
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch)) # 总迭代次数/每个epoch的迭代次数=总epoch数
            logger.info('Training statistics:' 
                        f'\n\tNumber of train images: {len(train_set)}' # 训练集图片数量
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}' # 数据集扩大比例
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}' # 每个GPU的批量大小
                        f'\n\tWorld size (gpu number): {opt["world_size"]}' # GPU数量
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}' # 每个epoch的迭代次数
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.') # 总epoch数，总迭代次数
        elif phase.split('_')[0] == 'val': # 验证集
            val_set = build_dataset(dataset_opt) # 从验证数据集中采样数据
            val_loader = build_dataloader( # 创建数据加载器
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed']) # 数据集，数据集选项，GPU数量，分布式训练，采样器，随机种子
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}') # 验证集图片/文件夹数量
            val_loaders.append(val_loader) # 将验证数据加载器添加到列表中
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt): # 加载恢复状态
    resume_state_path = None # 初始化
    if opt['auto_resume']: # 函数会尝试根据实验名称自动查找恢复状态文件
        state_path = osp.join('experiments', opt['name'], 'training_states') #检查获取的路径
        if osp.isdir(state_path): # 如果包含目标文件的目录存在
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False)) # 扫描目录
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state') #数从文件名中提取迭代编号，并选择具有最高（最新）迭代编号的状态文件
                opt['path']['resume_state'] = resume_state_path #该选定状态文件的路径存储在resume_state_path
    else:
        if opt['path'].get('resume_state'): # 如果存在resume_state
            resume_state_path = opt['path']['resume_state'] #将使用配置中的值覆盖路径

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device() #获取当前CUDA设备的ID,并将其分配给device_id
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id)) #torch.load加载状态文件，map_location将存储的张量映射到当前 GPU 设备
        check_resume(opt, resume_state['iter']) #对恢复状态执行额外的检查
    return resume_state

def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True) #解析训练选项,加载并处理YAML配置文件以获得各种训练设置和选项
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True # 为GPU加速
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt) #加载任何可用的恢复状态。恢复状态通常包含有关模型检查点和训练进度的信息
    # mkdir for experiments and logger
    if resume_state is None: #如果没有找到恢复状态，则创建新的实验目录
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log") # 日志文件路径
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)# get_root_logger函数用于创建记录器,称为“basicsr”
    logger.info(get_env_info()) # 获取环境信息,例如可用的 GPU、Python 版本和系统信息
    logger.info(dict2str(opt)) # 将'opt'转换为字符串并记录
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)
    # initialize wandb
    # if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')is not None):
    #     wandb.init(project=opt['logger']['wandb']['project'], name=opt['name'], config=opt)
    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt) # 用于创建深度学习模型的函数。会读取模型配置并相应地实例化模型架构
    if resume_state:  # resume training #表示训练正在从上一个检查点恢复
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch'] #有关上次运行的训练纪元和迭代的信息
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger) # 用于记录训练信息的日志记录器

    # dataloader prefetcher # 数据加载器预取器
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode') # 获取预取数据的方法：'cpu'、'cuda'或None
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader) #预取器负责将数据预取到CPU内存。有助于从GPU上卸载一些数据加载操作，从而提高训练效率
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt) #预取器负责将数据预取到GPU内存。有助于从GPU上卸载一些数据加载操作，从而提高训练效率
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1): # 遍历epoch
        train_sampler.set_epoch(epoch) #将训练采样器设置为当前纪元，这对于每个时期以不同顺序对数据进行洗牌和采样非常重要
        prefetcher.reset() # 重置预取器
        train_data = prefetcher.next() # 获取下一批数据

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1)) # 更新学习率
            # training
            model.feed_data(train_data) # 将数据加载到模型中
            model.optimize_parameters(current_iter) # 优化参数
            iter_timer.record() # 记录迭代时间
            if current_iter == 1: # 如果是第一次迭代
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time() # 重置开始时间以获得更准确的eta_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0: # 每隔一定迭代次数打印一次日志
                log_vars = {'epoch': epoch, 'iter': current_iter} #记录和报告训练信息和统计数据
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if (current_iter % opt['logger']['save_checkpoint_freq'] == 0) : # 每隔一定迭代次数保存一次模型和训练状态
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0) :#在验证数据集上运行模型
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start() #重置数据加载和迭代计时器以测量下一次迭代
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest #保存最新模型
    if opt.get('val') is not None:
        for val_loader in val_loaders: #对val_loaders中指定的每个验证数据集执行验证
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__': #直接运行脚本时执行代码块
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir)) #它通过从当前脚本文件的目录(__file__)向上导航两层(由sp.pardir表示)来计算根路径
    train_pipeline(root_path)  # 调用train_pipeline函数来启动训练过程