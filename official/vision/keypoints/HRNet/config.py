import megengine as mge
import  os

class Config():

    ## model settings
    base_channels = 32

    model_name = 'HRNet-W{}'.format(base_channels)
    
    block_type = {
        'stage1': BasicBlock,
        'stage2': Bottleneck,
        'stage3': Bottleneck,
        'stage4': Bottleneck
    }

    num_modules = {
        'stage1': 1,
        'stage2': 1,
        'stage3': 4,
        'stage4': 3
    }

    num_blocks = {
        'stage1': 4,
        'stage2': 4,
        'stage3': 4,
        'stage4': 4
    }

    ## dir
    save_dir = '/data/models'
    save_dir = os.path.join(args.save, args.arch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mge.set_log_file(os.path.join(save_dir, "log.txt"))


