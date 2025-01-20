# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
from tqdm import tqdm
import torch
from mmengine.model import revert_sync_batchnorm
from mmseg.apis.inference import _preprare_data
from mmseg.apis import init_model
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from mmengine.config import Config, DictAction
from mmengine.registry import MODELS
import json
from torchvision.transforms import CenterCrop
import torchinfo
from mmseg.models.backbones.sdtv3 import MultiSpike_norm

BASE_PATH = './work_dirs'
ADE20K = '/data/shzou/qingfeng/ADEChallengeData2016/images/validation'
VOC2012 = '/raid/ligq/lzx/data/VOCdevkit/VOC2012'


def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out_file', default='/data/shzou/qingfeng/Espikeformer/work_dir/out',
                        help='Path to output file')
    parser.add_argument('--in_file',
                        default=ADE20K,
                        help='Path to val file')
    parser.add_argument('--test_num',
                        default=100,
                        help='Number of testing images for cal the firing rate')
    parser.add_argument('--size',
                        default=512,  # 224
                        help='Image resolution of testing')
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
    # checkpoint = os.path.join(BASE_PATH+'/'+exp_name, 'iter_80000.pth')
    checkpoint = '/data/shzou/qingfeng/Espikeformer/ckpt/10M/iter_137500.pth'
    # checkpoint = '/home/shzou2/qfli/effcient_qtrick/tools/work_dirs/fpn_sdtv3_512x512_10M_ade20k/iter_160000.pth'
    # checkpoint = '/data/shzou/qingfeng/Espikeformer/ckpt/19M/iter_56000.pth'

    model = init_model(cfg, checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    print("Successful Build model.")

    functional.reset_net(model)
    firing_dict = {}

    # model_dict = {}
    # def forward_hook_fn(module, input, output):  # 计算每一层的发放率
    #     global i
    #     if module.name == 'MultiSpike_norm':
    #         i = i+1
    #     x_shape = torch.tensor(list(output.shape))
    #     z = torch.nonzero(output)
    #     all_neural = torch.prod(x_shape)
    #     if module.name  not in fr_dict.keys():
    #         fr_dict[module.name] = output.detach().mean() / iter
    #     else:
    #         fr_dict[module.name] = fr_dict[module.name] + output.detach().mean() / iter

    def forward_hook_fn(module, input, output):
        # 对 backbone 部分的T4 lif 单独处理
        # if output.shape[0] == 8:
        #     output = output.mean(0)
        output = output.mean(0)
        firing_dict[module.name] = output[-1].detach()
        # breakpoint()

    # # 对于 fpn neck 需要分层进行平均
    # def operate(value):
    #     return value.mean(0)

    # output_new = []
    # def forward_hook_fn(module, input, output):
    #     # if output.shape[0] == 8:
    #     #     output = output.mean(0)
    #     for out in output:
    #         new_out = operate(out)
    #         output_new.append(new_out)
    #     firing_dict[module.name] = output_new[-1].detach()
    #     # breakpoint()

    # 对于 backbone 部分 ，直接进行求均值即可
    # for n, m in model.named_modules():
    #     if isinstance(m,MultiSpike_norm):
    #         m.name = n
    #         m.register_forward_hook(forward_hook_fn)
    #         # if n == "scale_heads.0.0.1":
    #         # breakpoint()

    for n, m in model.decode_head.named_modules():
        m.name = n
        m.register_forward_hook(forward_hook_fn)

    # for n, m in model.neck.named_modules():
    #     m.name = n
    #     m.register_forward_hook(forward_hook_fn)

    # init the firing_dict
    T = getattr(model, "T", 1)

    fr_dict, nz_dict = {}, {}
    for i in range(T):
        fr_dict["t" + str(i)] = {}
        nz_dict["t" + str(i)] = {}
        # breakpoint()

    if "VOCdevkit" in args.in_file:
        imgs = []
        with open(os.path.join(args.in_file, 'ImageSets/Segmentation/val.txt'), 'r') as f:
            for line in f:
                imgs.append(os.path.join(args.in_file, 'JPEGImages', line.strip() + '.jpg'))
    else:
        imgs = os.listdir(args.in_file)

    test_num = args.test_num
    imgs = imgs[:test_num]
    last_idx = test_num - 1
    for i in tqdm(range(len(imgs)), desc="Testing:"):
        img = imgs[i]
        img = os.path.join(args.in_file, img)
        data, is_batch = _preprare_data(img, model)
        data_preprocessor = cfg.data_preprocessor
        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        data = data_preprocessor(data, False)

        # print(last_idx)
        with torch.no_grad():
            images = data['inputs'].to(args.device, non_blocking=True)
            images = CenterCrop(size=args.size)(images)

            output = model(images)

            functional.reset_net(model)

            # breakpoint()
            for t in range(T):
                fr_single_dict = calc_firing_rate(
                    firing_dict, fr_dict["t" + str(t)], last_idx, t
                )
                nz_single_dict = calc_non_zero_rate(
                    firing_dict, nz_dict["t" + str(t)], last_idx, t
                )
                nz_dict["t" + str(t)] = nz_single_dict
                fr_dict["t" + str(t)] = fr_single_dict
            # firing_dict = {}
            # import pdb; pdb.set_trace()

    # del firing_dict
    # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    non_zero_str = json.dumps(nz_dict, indent=4)
    firing_rate_str = json.dumps(fr_dict, indent=4)
    print("non-sero rate: ")
    print(non_zero_str)
    print("\n firing rate: ")
    print(firing_rate_str)
    import pandas as pd
    fr_dict = pd.DataFrame(fr_dict)
    if not os.path.exists(os.path.join(args.out_file, exp_name)):
        os.makedirs(os.path.join(args.out_file, exp_name))
    fr_dict.to_csv(os.path.join(args.out_file, exp_name, '10M_T8_decode_head.csv'))

    exit(0)


def calc_firing_rate(s_dict, fr_dict, idx, t):
    for k, v_ in s_dict.items():
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


if __name__ == '__main__':
    main()
