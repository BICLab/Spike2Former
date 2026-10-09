# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
from tqdm import tqdm
import torch
from mmengine.model import revert_sync_batchnorm
from mmengine.dataset import Compose

from mmdet.apis import init_detector, inference_detector
from functools import partial
from mmdet.models.utils.Qtrick import MultiSpike_norm4, MultiSpike_4
from mmdet.models.backbones.sdtv3 import MultiSpike_norm
# from mmdet.models.backbones.sdtv2 import MultiSpike_norm4
from mmdet.structures.det_data_sample import DetDataSample
import pandas as pd
from mmengine.config import Config, DictAction
from mmengine.registry import MODELS
# from analysis_tools.profile import profile
import json
# from mmseg.structures.seg_data_sample import SegDataSample
from torchvision.transforms import CenterCrop
import torch
import torchinfo
import torchvision
# from thop import profile
from calflops import calculate_flops
# from analysis_tools.profile import profile
from mmengine.analysis import get_model_complexity_info
# from torch_flops import TorchFLOPsByFX
from fvcore.nn import FlopCountAnalysis, parameter_count_table

BASE_PATH = './work_dirs'
ADE20K = os.getenv('ADE20K_VAL_IMAGES', 'data/ade20k/images/validation')
VOC2012 = os.getenv('VOC2012_ROOT', 'data/VOCdevkit/VOC2012')
CityScape = os.getenv('CITYSCAPES_ROOT', 'data/cityscapes/')
COCO = os.getenv('COCO_VAL_IMAGES', 'data/coco/val2017/')
from mmdet.utils import ConfigType

def build_test_pipeline(cfg: ConfigType) -> ConfigType:
    """Build test_pipeline for mot/vis demo. In mot/vis infer, original
    test_pipeline should remove the "LoadImageFromFile" and
    "LoadTrackAnnotations".

    Args:
         cfg (ConfigDict): The loaded config.
    Returns:
         ConfigType: new test_pipeline
    """
    # remove the "LoadImageFromFile" and "LoadTrackAnnotations" in pipeline
    transform_broadcaster = cfg.test_dataloader.dataset.pipeline
    # import pdb; pdb.set_trace()

    test_pipeline = Compose(transform_broadcaster)

    return test_pipeline

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out_file', default='./work_dirs',
                        help='Path to output file')
    parser.add_argument('--in_file',
                        default=COCO,
                        help='Path to val file')
    parser.add_argument('--get_flops',
                        default=True,
                        help='Calculate the flops')
    parser.add_argument('--painting',
                        default=True,
                        help='print the model layers')
    parser.add_argument('--test_num',
                        default=200,
                        help='Number of testing images for cal the firing rate')
    parser.add_argument('--size',
                        default=(1024, 1024),
                        help='Image resolution of testing')
    parser.add_argument('--quant',
                        default=4,
                        help='normalization')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--print_model', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
             'If specified, it will be automatically saved '
             'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # build the model from a config file and a checkpoint file
    exp_name = args.config.split('/')[3].split('.')[0]
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None
    # checkpoint = None

    print('==> Building model..')

    print(cfg)
    model = init_detector(cfg, checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    print("Successful Build model.")

    firing_dict = {}
    quant = args.quant
    # import pdb; pdb.set_trace()
    def forward_hook_fn(module, input, output):
        if output.detach().dtype == torch.float32:
            firing_dict[module.name] = output.detach() * args.quant
        else:
            firing_dict[module.name] = output.detach()

    for n, m in model.named_modules():
        if isinstance(m, MultiSpike_norm):  # for Backbone
            m.name = n
            m.register_forward_hook(forward_hook_fn)
        if isinstance(m, MultiSpike_4):  # for maskformerhead
            m.name = n
            m.register_forward_hook(forward_hook_fn)
        if isinstance(m, MultiSpike_norm4):  # for Others
            m.name = n
            m.register_forward_hook(forward_hook_fn)

    # init the firing_dict
    T = getattr(model, "T", 4)
    fr_dict, nz_dict = {}, {}
    for i in range(T):
        fr_dict["t" + str(i)] = {}
        nz_dict["t" + str(i)] = {}

    # import pdb; pdb.set_trace()
    # print(args)
    if "VOCdevkit" in args.in_file:
        imgs = []
        with open(os.path.join(args.in_file, 'ImageSets/Segmentation/val.txt'), 'r') as f:
            for line in f:
                imgs.append(os.path.join(args.in_file, 'JPEGImages', line.strip() + '.jpg'))
    elif "cityscapes" in args.in_file:
        imgs = []
        with open(os.path.join(args.in_file, 'val.txt'), 'r') as f:
            for line in f:
                imgs.append(os.path.join(args.in_file, 'leftImg8bit/val', line.strip() + '_leftImg8bit.png'))
    else:
        imgs = os.listdir(args.in_file)

    test_num = args.test_num
    imgs = imgs[:test_num]
    last_idx = test_num - 1
    flag = 1
    for i in tqdm(range(len(imgs)), desc="Testing:"):
        img = imgs[i]
        img = os.path.join(args.in_file, img)
        # import pdb; pdb.set_trace()
        test_pipline = build_test_pipeline(cfg)
        data = inference_detector(model, img, test_pipeline=test_pipline)
        data_preprocessor = cfg.data_preprocessor
        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        data = data_preprocessor(data, False)

        # prepare data metainfo for input
        batch_img_metas = [
            data_sample.metainfo for data_sample in data['data_samples']
        ]
        batch_data_samples = []
        for metainfo in batch_img_metas:
            metainfo['batch_input_shape'] = metainfo['img_shape']
            # import pdb; pdb.set_trace()
            batch_data_samples.append(DetDataSample(metainfo=metainfo))

        # print(last_idx)
        with torch.no_grad():
            images = data['inputs'].to(args.device, non_blocking=True)
            images = CenterCrop(size=args.size)(images)

            output = model.predict(images, data['data_samples'])
            if args.get_flops and flag == 1:
                # flops, params, layer_info = profile(model, (images, batch_data_samples), ret_layer_info=True)
                # # flops, params = profile(model, images)  FPN
                # print('Profile: Flops: %.2f G, Params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
                _forward = model.forward
                model.forward = partial(_forward, data_samples=batch_data_samples)
                flops, macs, params = calculate_flops(model=model,
                                                      args=[images],
                                                      print_results=False,
                                                      print_detailed=False)
                print("Calflops: FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
                flops = FlopCountAnalysis(model, images)
                print("FVCORE: InputShape:%s FLOPs:%.2f G " % (str(images.shape), flops.total() / 1000000000.0))


                flag = 0

            firing_dict = {key: expand_with_binary_sequence(value, 4) for key, value in firing_dict.items()}
            for t in range(T):
                fr_single_dict = calc_firing_rate(
                    firing_dict, fr_dict["t" + str(t)], last_idx, t
                )
                fr_dict["t" + str(t)] = fr_single_dict
                nz_single_dict = calc_non_zero_rate(
                    firing_dict, nz_dict["t" + str(t)], last_idx, t
                )
                nz_dict["t" + str(t)] = nz_single_dict
            firing_dict = {}

    del firing_dict

    non_zero_str = json.dumps(nz_dict, indent=4)
    firing_rate_str = json.dumps(fr_dict, indent=4)
    print("non-zero rate: ")
    print(non_zero_str)
    print("\n firing rate: ")
    print(firing_rate_str)

    if not os.path.exists(os.path.join(args.out_file, exp_name)):
        os.makedirs(os.path.join(args.out_file, exp_name))
    print('Saving firing rate to: ', os.path.join(args.out_file, exp_name, 'fr_rate.csv'))
    fr_dict = pd.DataFrame(fr_dict)
    fr_dict.to_csv(os.path.join(args.out_file, exp_name, 'fr_rate.csv'), )

    exit(0)

    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     title=args.title,
    #     opacity=args.opacity,
    #     draw_gt=False,
    #     show=False if args.out_file is not None else True,
    #     out_file=args.out_file)


def calc_firing_rate(s_dict, fr_dict, idx, t):
    for k, v_ in s_dict.items():
        # import pdb; pdb.set_trace()
        v = v_[t, ...]
        if k in fr_dict.keys():
            fr_dict[k] += v.mean().item() / idx
        else:
            fr_dict[k] = v.mean().item() / idx

    return fr_dict


def calc_non_zero_rate(s_dict, nz_dict, idx, t):
    for k, v_ in s_dict.items():
        v = v_[t, ...]
        x_shape = torch.tensor(list(v.shape))
        all_neural = torch.prod(x_shape)
        z = torch.nonzero(v)
        if k in nz_dict.keys():
            nz_dict[k] += (z.shape[0] / all_neural).item() / idx
        else:
            nz_dict[k] = (z.shape[0] / all_neural).item() / idx
    return nz_dict


def expand_with_binary_sequence(x, N):
    input_shape = x.shape
    expanded_tensor = torch.zeros((*input_shape, N), dtype=torch.float, device='cuda')
    indices = torch.arange(N, device='cuda').unsqueeze(0).expand(input_shape + (N,)).contiguous()
    expanded_tensor[indices < x.unsqueeze(-1)] = 1
    if len(input_shape) == 5:
        expanded_tensor = expanded_tensor.permute(0, 5, 1, 2, 3, 4)
    elif len(input_shape) == 4:
        expanded_tensor = expanded_tensor.permute(0, 4, 1, 2, 3)
    elif len(input_shape) == 3:
        expanded_tensor = expanded_tensor.permute(0, 3, 1, 2)
    expanded_tensor = expanded_tensor.transpose(0, 1)
    # expanded_tensor = expanded_tensor.flatten(0,1)
    return expanded_tensor


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():

        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # 构造新的键
        if isinstance(v, tuple) and isinstance(v[2], dict):
            # 如果v是一个元组且v的第三个元素是字典，递归处理这个字典
            if v[2]:  # 如果内嵌字典不为空，继续递归
                items.extend(flatten_dict(v[2], new_key, sep=sep).items())
            # 还要添加当前层的参数和flops
            items.append((new_key, (v[0], v[1])))
        else:
            items.append((new_key, v))  # 如果不是字典，直接添加
    return dict(items)


def get_Conv_spike_dict(d: dict):
    out_dict = {}
    for k, v in d.items():
        if 'bn' not in k:
            out_dict[k] = v
    return out_dict


if __name__ == '__main__':
    main()
