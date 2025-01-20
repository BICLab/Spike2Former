# CONFIG=../configs/Spikeformer/SDTv2_maskformer_DCNpixelDecoder_ade20k.py
CONFIG=../configs/Spikeformer/SDTv2_maskformer_DCNPixelDecoder_CityScapes.py
# CONFIG=../configs/Spikeformer/SDTv2_Spike2former_voc_512x512.py
# CHECKPOINT='/public/liguoqi/lzx/code/mmseg/tools/work_dirs/SDTv2_maskformer_DCNpixelDecoder_ade20k/best_mIoU_iter_102500.pth' # ADE20k Train 46.3 Test 44.5 ()
# CHECKPOINT='/public/liguoqi/lzx/code/mmseg/tools/work_dirs/Ablation/v2_spike2former_voc2012_1x4/best_mIoU_iter_97500.pth' # VOC2012 Train 76.3 Test (Q_IFNode)
# CHECKPOINT='/public/liguoqi/lzx/code/mmseg/tools/work_dirs/V2_Spike2former_withoutshortcut/best_mIoU_iter_152500.pth'  # ADE20k 1x4 without shortcut
CHECKPOINT='/public/liguoqi/lzx/code/mmseg/tools/work_dirs/SDTv2_maskformer_DCNPixelDecoder_CityScapes/best_mIoU_iter_47500.pth'  # CityScapes 74.2 (Multi-Spikenorm)

GPUS=1

PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4} \
    --out ./work_dir/vis_result/CityScapes\
    --show-dir ./work_dirs/vis_result/CityScapes/1x4
