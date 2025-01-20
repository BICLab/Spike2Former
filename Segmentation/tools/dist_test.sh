#CONFIG=$1
#GPUS=$2
#CHECKPOINT='/home/zxlei/mmseg/tools/work_dirs/fpn_SDT_512x512_512_ade20k/iter_160000.pth'
##in_file=$3
#PORT=${PORT:-29500}
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch \
#    --nproc_per_node=$GPUS \
#    --master_port=$PORT \
#    $(dirname "$0")/test.py \
#    $CHECKPOINT \
#    $CONFIG \
#    --launcher pytorch \
#    ${@:4}

#CONFIG=../configs/spike2former/SDTv2_maskformer_DCNpixelDecoder.py
#CONFIG=../configs/Spikeformer/SDTv3_b_Spike2former_ade20k_512x512.py
# CONFIG=../configs/Spikeformer/SDTv2_maskformer_DCNpixelDecoder_ade20k.py
#CONFIG=../configs/Spikeformer/SDTv3_b_Spike2former_Cityscapes_512x1024.py
#CONFIG=../configs/Spikeformer/SDTv2_maskformer_DCNPixelDecoder_CityScapes.py
CONFIG=../configs/Spikeformer/SDTv2_Spike2former_voc_512x512.py
#CONFIG=../configs/sem_sdt/fpn_sdtv3_512x512_19M_ade20k.py
#CHECKPOINT=/home/zxlei/SF/tools/work_dirs/SDTv2_maskformer_DCNpixelDecoder/iter_150000.pth
#CHECKPOINT=/raid/ligq/lzx/efficient_snn/segmentation/tools/work_dirs/fpn_sdtv3_512x512_10M_ade20k
#CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/SDTv3_19M_Spike2former_ADE20k_512x512/best_mIoU_iter_140000.pth
#CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/SDTv2_15M_Spike2former_ADE20k_512x512/best_mIoU_iter_105000.pth
#CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/SDTv3_b_Spike2former_Cityscapes_512x1024/best_mIoU_iter_80000.pth
#CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/SDTv2_maskformer_DCNPixelDecoder_CityScapes/best_mIoU_iter_47500.pth
#CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/SDTv2_Spike2former_voc_512x512/best_mIoU_iter_25000.pth
#CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/Ablation/v2_spike2former_voc2012_1x4/best_mIoU_iter_97500.pth # 1x4
# CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/Ablation/v2_spike2former_voc2012_4x4/best_mIoU_iter_107500.pth # 4x4
#CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/Ablation/v2_spike2former_voc2012_2x2/best_mIoU_iter_115000.pth # 2x2
#CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/Ablation/v2_spike2former_voc2012_1x2/best_mIoU_iter_92500.pth # 1x2
# CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/V2_Spike2former_withoutshortcut/best_mIoU_iter_152500.pth
CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/V2_Spike2former_VOC2x4/best_mIoU_iter_25000.pth  # VOC 2x4
# CHECKPOINT=/public/liguoqi/lzx/code/mmseg/tools/work_dirs/V2_Spike2former_VOC1x8/best_mIoU_iter_27500.pth  # VOC 1x8
GPUS=1
PORT=${PORT:-29400}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/cal_firing_num.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --show-dir ./work_dirs/timestamp/show_dir/V2_ade \
    ${@:4}
