CONFIG=$1
GPUS=$2
PORT=${PORT:-24834}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}  \
    --work-dir ./work_dirs/Final/V2_Spike2former_CityScape
#    --work-dir ./work_dirs/Ablation/v2_spike2former_voc2012_1x2 \
#    --resume
#    --work-dir ./work_dirs/v2_spike2former_voc2012_2x2


#    --work-dir ./work_dirs/Verify_Cross_Attention

#    --work-dir ./work_dirs/v2_spike2former_voc2012_4x4
#    --resume \
#    --work-dir ./work_dirs/Ablation_normalized_sigmoid_after \