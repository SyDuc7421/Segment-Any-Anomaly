"""Factory dựng SAM predictor và saliency extractor theo tên.

Mọi import nặng đều nằm trong thân hàm, có chủ đích: thiếu MobileSAM thì
chỉ nhánh MobileSAM hỏng, nhánh baseline vit_h vẫn chạy được.
"""

SAM_VARIANTS = {
    'vit_h': 'SAM ViT-H goc, 2.4 GB — baseline',
    'mobile_sam': 'MobileSAM (vit_t)',
    'efficientvit_l0': 'EfficientViT-SAM-L0',
}

SALIENCY_BACKBONES = {
    'wide_resnet50': 'wide_resnet50_2',
    'mobilenetv3': 'mobilenetv3_large_100',
}


def build_sam_predictor(variant, checkpoint, device):
    """Trả về đối tượng có interface SamPredictor: set_image, transform, predict_torch."""
    if variant not in SAM_VARIANTS:
        raise ValueError(
            f"Unknown sam_variant '{variant}'. Valid names: {sorted(SAM_VARIANTS)}"
        )

    if variant == 'vit_h':
        from SAM.segment_anything import SamPredictor, build_sam

        return SamPredictor(build_sam(checkpoint=checkpoint).to(device))

    if variant == 'mobile_sam':
        from mobile_sam import SamPredictor, sam_model_registry

        sam = sam_model_registry['vit_t'](checkpoint=checkpoint)
        sam.to(device)
        sam.eval()
        return SamPredictor(sam)

    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from efficientvit.sam_model_zoo import create_sam_model

    sam = create_sam_model(name='l0', pretrained=True, weight_url=checkpoint)
    sam = sam.to(device).eval()
    return EfficientViTSamPredictor(sam)


def build_saliency_extractor(name, device):
    """Trả về ModelINet dùng backbone tương ứng."""
    if name not in SALIENCY_BACKBONES:
        raise ValueError(
            f"Unknown saliency_backbone '{name}'. Valid names: {sorted(SALIENCY_BACKBONES)}"
        )

    from .modelinet import ModelINet

    return ModelINet(
        device=device,
        backbone_name=SALIENCY_BACKBONES[name],
        out_indices=(1, 2, 3),
    )
