# Copyright (c) OpenMMLab. All rights reserved.
from .activations import SiLU
from .brick_wrappers import AdaptiveAvgPool2d, adaptive_avg_pool2d
from .conv_upsample import ConvUpsample
from .dropblock import DropBlock
from .matrix_nms import mask_matrix_nms
from .msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from .normed_predictor import NormedConv2d, NormedLinear
from .pixel_decoder import PixelDecoder, TransformerEncoderPixelDecoder,  DCNTransformerEncoderPixelDecoder
from .positional_encoding import (LearnedPositionalEncoding,
                                  SinePositionalEncoding,
                                  SinePositionalEncoding3D,
                                  PositionEmbeddingSine)
# yapf: disable
from .transformer import (MLP, AdaptivePadding,
                          DeformableDetrTransformerDecoder,
                          DeformableDetrTransformerDecoderLayer,
                          DeformableDetrTransformerEncoder,
                          DeformableDetrTransformerEncoderLayer,
                          DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer,
                          Mask2FormerTransformerDecoder,
                          Mask2FormerTransformerDecoderLayer,
                          Mask2FormerTransformerEncoder,
                          PatchEmbed,
                          Spike2FormerTransformerEncoder,
                          Spike2FormerTransformerDecoderLayer,
                          SpikeMask2FormerTransformerDecoder,
                          PatchMerging, coordinate_to_encoding,
                          inverse_sigmoid, nchw_to_nlc, nlc_to_nchw)

# yapf: enable

__all__ = [
    'mask_matrix_nms', 'DropBlock',
    'PixelDecoder', 'TransformerEncoderPixelDecoder',
    'MSDeformAttnPixelDecoder', 'PatchMerging',
    'SinePositionalEncoding', 'LearnedPositionalEncoding',
    'NormedLinear', 'NormedConv2d',
    'ConvUpsample', 'adaptive_avg_pool2d',
    'AdaptiveAvgPool2d', 'PatchEmbed', 'nchw_to_nlc', 'nlc_to_nchw',
    'inverse_sigmoid', 'SiLU', 'MLP',
    'DetrTransformerEncoderLayer', 'DetrTransformerDecoderLayer',
    'DetrTransformerEncoder', 'DetrTransformerDecoder',
    'DeformableDetrTransformerEncoder', 'DeformableDetrTransformerDecoder',
    'DeformableDetrTransformerEncoderLayer',
    'DeformableDetrTransformerDecoderLayer', 'AdaptivePadding',
    'coordinate_to_encoding',
    'Mask2FormerTransformerEncoder',
    'Mask2FormerTransformerDecoderLayer', 'Mask2FormerTransformerDecoder',
    'SinePositionalEncoding3D',
    "DCNTransformerEncoderPixelDecoder"
    # 'SpikeMSDeformAttnPixelDecoder',
]
