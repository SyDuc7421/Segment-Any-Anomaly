"""Factory detector open-vocabulary.

Grounding DINO khong di qua day: no duoc dung truc tiep trong SAA/model.py qua
load_dino, va duong do phai giu nguyen tung dong de baseline tai lap bit-exact
(spec muc 3). File nay chi phuc vu cac detector tra ve diem theo tung query.

Khac biet cot loi giua hai loai:

    Grounding DINO  tra ve ma tran (so query x 256 token BERT). Pipeline hoi
                    nguoc "box nay khop chu nao trong cau" de loc nen.
    Query-scored    tra ve mot diem moi (box, query). "Box nay khop cum nao"
                    la argmax theo query.

Moi import nang nam trong than ham, co chu dich: thieu ultralytics thi chi
hong nhanh yolo_world chu khong hong baseline, va test nap duoc file nay tren
may khong co torch.
"""

DETECTORS = {
    'grounding_dino': 'Grounding DINO Swin-T — baseline, dung qua load_dino',
    'yolo_world': 'YOLO-World — open-vocab thoi gian thuc',
    'owlv2': 'OWLv2 base patch16 ensemble — qua transformers',
}

# Loai tra diem moi (box, query), di qua Model.query_detector_proposal.
QUERY_SCORED = {'yolo_world', 'owlv2'}

DEFAULT_DETECTOR_WEIGHTS = {
    'yolo_world': 'yolov8s-worldv2.pt',
    'owlv2': 'google/owlv2-base-patch16-ensemble',
}


def split_phrase(phrase):
    """Tach mot cau prompt thanh cac cum rieng.

    Prompt trong repo co dang 'blue defect. black defect. scratch.'. Grounding
    DINO nuot ca cau va cham diem theo token; detector query-scored can tung
    cum lam mot class rieng.

    Giu nguyen quirk dau phay o spec muc 2.4 ('carpet,'): do la hanh vi
    baseline, khong phai loi can sua o day.
    """
    return [term.strip() for term in phrase.split('.') if term.strip()]


def xyxy_to_cxcywh_norm(boxes, width, height):
    """Doi box pixel xyxy sang cxcywh chuan hoa [0, 1].

    Phia sau pipeline tinh dien tich bang `boxes[:, 2] * boxes[:, 3]` roi moi
    denormalize, nen tra sai he toa do la loi im lang: dien tich sai keo theo
    defect_max_area sai, ma anomaly map van ra so nhin co ve hop ly.
    """
    out = []
    for x_min, y_min, x_max, y_max in boxes:
        box_w = (x_max - x_min) / width
        box_h = (y_max - y_min) / height
        cx = (x_min + x_max) / 2 / width
        cy = (y_min + y_max) / 2 / height
        out.append((cx, cy, box_w, box_h))
    return out


class YoloWorldDetector:
    """YOLO-World qua ultralytics."""

    def __init__(self, weights, device):
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(weights)
        self.device = device

    def detect(self, image_bgr, terms, score_thr):
        """Tra ve (boxes_cxcywh_norm, scores, phrases) cho mot danh sach cum."""
        self.model.set_classes(terms)
        results = self.model.predict(
            image_bgr, conf=score_thr, device=self.device, verbose=False
        )

        result = results[0]
        height, width = image_bgr.shape[:2]

        xyxy = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        phrases = [terms[int(i)] for i in result.boxes.cls.cpu().tolist()]

        return xyxy_to_cxcywh_norm(xyxy, width, height), scores, phrases


class Owlv2Detector:
    """OWLv2 qua transformers. Khong them dependency moi."""

    def __init__(self, weights, device):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(weights)
        self.model = Owlv2ForObjectDetection.from_pretrained(weights).to(device).eval()

    def detect(self, image_bgr, terms, score_thr):
        import cv2
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        height, width = image_bgr.shape[:2]

        inputs = self.processor(text=[terms], images=pil, return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([[height, width]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=score_thr
        )[0]

        xyxy = results['boxes'].cpu().tolist()
        scores = results['scores'].cpu().tolist()
        phrases = [terms[int(i)] for i in results['labels'].cpu().tolist()]

        return xyxy_to_cxcywh_norm(xyxy, width, height), scores, phrases


def build_detector(name, device, weights=None):
    """Dung mot detector query-scored.

    grounding_dino KHONG di qua day - xem docstring dau file.
    """
    if name not in DETECTORS:
        raise ValueError(
            f"Unknown detector '{name}'. Valid names: {sorted(DETECTORS)}"
        )

    if name == 'grounding_dino':
        raise ValueError(
            "grounding_dino khong dung qua build_detector: SAA/model.py dung no "
            "truc tiep bang load_dino de giu nguyen duong baseline."
        )

    weights = weights or DEFAULT_DETECTOR_WEIGHTS[name]

    if name == 'yolo_world':
        return YoloWorldDetector(weights, device)

    return Owlv2Detector(weights, device)
