"""Sinh prompt spec bang VLM trong so mo chay cuc bo. Khong goi API ngoai.

RANH GIOI RO RI DU LIEU (spec muc 5.5): script nay chi duoc doc `train` split.
train_image_paths hard-code duong dan `train/good` va co test chan moi duong
khac. Khong cham anh test, khong cham ground-truth, khong cham anh anomaly.

RO RI KIEU THU HAI, de bo sot hon: prompt he thong KHONG duoc chua phat hien do
tren tap test. Thang so sanh cho thay class texture hop voi prompt generic hon
(results/phase_a_ladder/README.md), nhung nhet dieu do vao day la ma hoa ket qua
tap test vao bo sinh prompt - P2 mat tu cach zero-shot. Chi duoc noi su that ve
CHI PHI (moi prompt ton mot luot detector), khong duoc noi su that ve KET QUA.
Co test chan dieu nay.

TAI LAP: greedy decoding (do_sample=False) tren trong so co dinh, nen cung
MODEL_ID + cung prompt cho ra cung mot file JSON.
"""

import argparse
import json
import os
import re

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'

SPLIT_DIRS = {
    # Chi train/good. Cay thu muc cua ca hai dataset deu theo dang nay.
    'mvtec': os.path.join('{class_name}', 'train', 'good'),
    'visa_public': os.path.join('{class_name}', 'train', 'good'),
}

SYSTEM = """You write text prompts for an open-vocabulary object detector \
(Grounding DINO) that finds manufacturing defects.

The detector grounds short noun phrases in an image. Phrases naming a visible \
defect appearance work; abstract quality judgements do not. Prefer two-to-three \
word phrases naming what the defect looks like, not what caused it.

Each prompt you emit costs one detector forward pass per image, so a short list \
of precise phrases is cheaper than a long list, and a phrase that grounds \
nothing costs the same as one that works.

Reply with one JSON object and nothing else."""

USER_TEMPLATE = """Industrial anomaly detection, object category: {class_name}

Produce a JSON object with exactly these keys:

- "object_prompt": the noun the detector uses to find the object itself. Bare \
noun, no article, no punctuation.
- "object_number": integer, how many instances of that object appear in one \
image. Usually 1.
- "k_mask": integer, how many candidate defect regions to keep per image. Use 5.
- "defect_area_threshold": float in (0, 1], the largest fraction of the object's \
area one defect may occupy. Use 0.9.
- "defect_prompts": a list of 1 to 5 objects, each with "text" (the phrase given \
to the detector) and "filter" (a phrase meaning the object itself; boxes matching \
it are dropped as background, so normally the same as object_prompt).

Choose how many defect prompts this category needs. Emit only phrases you expect \
the detector to ground in a real image of this category.

Reply with the JSON object only."""


def train_image_paths(dataset, class_name, root, limit):
    """Duong dan anh trong train/good, da sap xep.

    CHI train split. Day la ranh gioi chong ro ri du lieu o spec muc 5.5.
    Sap xep de greedy decoding tai lap duoc.
    """
    pattern = SPLIT_DIRS[dataset].format(class_name=class_name)
    directory = os.path.join(root, pattern)

    if not os.path.isdir(directory):
        return []

    names = sorted(n for n in os.listdir(directory)
                   if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))

    return [os.path.join(directory, n) for n in names[:limit]]


def build_messages(class_name, n_images):
    """Chat message. Anh dat TRUOC text; anh that truyen rieng cho processor."""
    content = [{'type': 'image'} for _ in range(n_images)]
    content.append({'type': 'text', 'text': USER_TEMPLATE.format(class_name=class_name)})

    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
        {'role': 'user', 'content': content},
    ]


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _llm_prompts():
    """Nap SAA/prompts/llm_prompts.py theo duong dan.

    `from SAA.prompts.llm_prompts import ...` se chay SAA/__init__.py, file do
    import .model va keo theo torch. Script nay khong can torch cho phan validate
    - va test cua no chay tren venv khong co torch.
    """
    import importlib.util

    path = os.path.join(_REPO_ROOT, 'SAA', 'prompts', 'llm_prompts.py')
    spec = importlib.util.spec_from_file_location('saa_llm_prompts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_spec(text, class_name):
    """Boc JSON ra khoi output model va validate.

    Model trong so mo thuong boc JSON trong ```json ... ``` hoac them loi dan,
    khac model co structured output cung buoc. Lay khoi { ... } dai nhat.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f'khong tim thay JSON trong output cho {class_name}:\n{text[:400]}')

    try:
        spec = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f'JSON hong cho {class_name}: {e}\n{match.group(0)[:400]}')

    spec['class'] = class_name
    _llm_prompts().validate_spec(spec)
    return spec


def build_model(load_in_4bit=True):
    """Nap VLM. Lop Auto* tu tra ra lop dung tu config, khong phai doan ten."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    kwargs = {'device_map': 'auto'}
    if load_in_4bit:
        # 7B fp16 la ~15 GB, sat tran 16 GB cua T4. 4-bit xuong ~5 GB.
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return processor, model


def generate_one(processor, model, class_name, image_paths, max_new_tokens=512, retries=3):
    """Sinh spec cho mot class, thu lai neu JSON hong.

    Greedy o moi lan, ke ca lan thu lai - nen chuoi thu lai cung tai lap duoc.
    """
    from PIL import Image

    images = [Image.open(p).convert('RGB') for p in image_paths]
    messages = build_messages(class_name, len(images))
    last_error = None
    last_output = ''

    for attempt in range(retries):
        if last_error is not None:
            messages = messages + [
                {'role': 'assistant', 'content': [{'type': 'text', 'text': last_output}]},
                {'role': 'user', 'content': [{'type': 'text', 'text':
                    f'That was rejected: {last_error}. Reply with the JSON object only.'}]},
            ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=images or None, return_tensors='pt'
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        last_output = processor.decode(
            output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )

        try:
            return parse_spec(last_output, class_name)
        except ValueError as e:
            last_error = str(e)
            print(f'  thu lai {attempt + 1}/{retries}: {last_error[:120]}')

    raise SystemExit(f'{class_name}: khong sinh duoc JSON hop le sau {retries} lan')


def main():
    parser = argparse.ArgumentParser(description='Sinh prompt spec bang VLM cuc bo')
    parser.add_argument('--dataset', choices=sorted(SPLIT_DIRS), required=True)
    parser.add_argument('--variant', choices=['blind', 'vision'], required=True)
    parser.add_argument('--data-root', required=True,
                        help='Thu muc goc dataset; chi train/good duoc doc')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-images', type=int, default=3,
                        help='So anh normal gui kem o variant vision')
    parser.add_argument('--fp16', action='store_true',
                        help='Nap fp16 thay vi 4-bit; can >16 GB VRAM')
    args = parser.parse_args()

    from datasets import dataset_classes

    processor, model = build_model(load_in_4bit=not args.fp16)
    specs = []

    for class_name in dataset_classes[args.dataset]:
        image_paths = []
        if args.variant == 'vision':
            image_paths = train_image_paths(
                args.dataset, class_name, args.data_root, args.n_images
            )
            if not image_paths:
                raise SystemExit(
                    f'khong tim thay anh train cho {class_name} trong {args.data_root} - '
                    f'variant vision can anh, dung im lang bo qua'
                )

        spec = generate_one(processor, model, class_name, image_paths)
        print(f"{class_name}: {len(spec['defect_prompts'])} prompt  "
              f"{[p['text'] for p in spec['defect_prompts']]}")
        specs.append(spec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)

    # Provenance canh file JSON: hoi dong tai sinh duoc chinh xac file nay.
    meta = {
        'model_id': MODEL_ID,
        'variant': args.variant,
        'dataset': args.dataset,
        'n_images': args.n_images if args.variant == 'vision' else 0,
        'decoding': 'greedy (do_sample=False)',
        'quantization': 'fp16' if args.fp16 else '4-bit nf4',
        'system_prompt': SYSTEM,
        'user_template': USER_TEMPLATE,
    }
    with open(args.out.replace('.json', '-meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    total = sum(len(s['defect_prompts']) for s in specs)
    print(f'\n{len(specs)} class, {total} prompt, trung binh '
          f'{total / len(specs):.1f} prompt/class -> {args.out}')
    print(f'So luot DINO moi anh se la 1 + so prompt; P3 thu cong dung 5.20 luot.')


if __name__ == '__main__':
    main()
