import json
import os
import subprocess
import sys
from datasets import dataset_classes

from SAA.backbones import DEFAULT_SAM_CHECKPOINTS
from utils.csv_utils import check_run_identity, completed_classes

if __name__ == '__main__':

    # Danh sach class chay hong. Truoc day day la Pool(processes=1) +
    # apply_async(os.system, ...), nuot sach ma tra ve: ca 15 class chet o
    # dong import van cho ra exit code 0 sau mot phut, nhin nhu chay xong.
    failures = []

    dataset_list = ['mvtec']
    gpu_indx = 0
    # ROOT_DIR cho phep tro thang vao Google Drive, de moi class chay xong la
    # an toan ngay thay vi cho toi cuoi lan chay moi copy ra. Cung la cach de
    # moi cau hinh Phase B co thu muc rieng, khong ghi de len nhau.
    root_dir = os.environ.get('ROOT_DIR') or './result'

    # Khong dat mac dinh. Spec muc 6.1: moi con so vao luan van phai chay full
    # test set, vi normalize() phu thuoc vao tap anh cua lan chay do.
    # `or None`: MAX_SAMPLES='' (bien rong) phai duoc coi nhu khong dat, khong
    # thi --max-samples se nhan chuoi rong va lam argparse loi.
    max_samples = os.environ.get('MAX_SAMPLES') or None

    # Spec muc 6.4: bat cal_pro chi cho cac cau hinh len bang chinh (baseline,
    # Lite thang cuoc, P2-vision, P3). r_f1 di chung co nay, nen tat co la cot
    # r_f1 rong. Mac dinh False vi PRO va r_f1 deu tinh tren CPU va cham.
    # `or 'False'`: cung ly do nhu MAX_SAMPLES o tren.
    cal_pro = os.environ.get('CAL_PRO') or 'False'

    # Anh truc quan hoa: ~117 anh moi class. Tat di khi ROOT_DIR nam tren Drive,
    # neu khong moi lan chay phai ghi hang nghin file nho qua Drive FUSE.
    # Mac dinh 'True' de giu nguyen hanh vi cu.
    vis = os.environ.get('VIS') or 'True'

    # Danh tinh cua lan chay nay - dung CHUNG mot dict de dung lenh command
    # line va de quyet dinh skip, tranh viec hai noi lech nhau. sam_variant
    # va saliency_backbone chua co CLI env o day (runner nay chi chay
    # baseline), nen khoa cung theo default cua eval_SAA.py.
    # Mac dinh la baseline. Buoc 2 cua spec muc 6.2 chay them cau hinh Lite,
    # dat SAM_VARIANT / SALIENCY_BACKBONE de doi - VA dat ROOT_DIR khac, vi
    # csv_path khong mang danh tinh cau hinh nen dung chung la ghi de len nhau.
    sam_variant = os.environ.get('SAM_VARIANT') or 'vit_h'
    saliency_backbone = os.environ.get('SALIENCY_BACKBONE') or 'wide_resnet50'
    detector = os.environ.get('DETECTOR') or 'grounding_dino'
    sam_checkpoint = os.environ.get('SAM_CHECKPOINT') or DEFAULT_SAM_CHECKPOINTS[sam_variant]

    # sam_variant va saliency_backbone NAM TRONG danh tinh: doi chung la con so
    # doi, nen ket qua cu khong con so sanh duoc va phai chay lai.
    run_identity = {
        'max_samples': int(max_samples) if max_samples else None,
        'cal_pro': cal_pro.lower() in ('yes', 'true', 't', '1'),
        'sam_variant': sam_variant,
        'saliency_backbone': saliency_backbone,
        'detector': detector,
    }

    for dataset in dataset_list:
        csv_path = os.path.join(root_dir, 'csv', f'{dataset}-indx-0.csv')
        meta_path = os.path.join(root_dir, 'csv', 'run_meta.json')

        done = completed_classes(csv_path)

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        classes = dataset_classes[dataset]
        for cls in classes[:]:
            meta_key = f'{dataset}-{cls}'

            if cls in done:
                ok, reason = check_run_identity(meta, meta_key, run_identity)
                if ok:
                    print(f'skip {cls}: da co ket qua trong {csv_path}')
                    continue
                print(f'CANH BAO: {cls} co ket qua trong CSV nhung KHONG dang '
                      f'tin ({reason}) - chay lai va ghi de.')

            cmd = [
                'python', 'eval_SAA.py',
                '--dataset', dataset,
                '--class-name', cls,
                '--batch-size', '1',
                '--root-dir', root_dir,
                '--cal-pro', str(run_identity['cal_pro']),
                '--sam-variant', run_identity['sam_variant'],
                '--saliency-backbone', run_identity['saliency_backbone'],
                '--sam_checkpoint', sam_checkpoint,
                '--detector', run_identity['detector'],
                '--gpu-id', str(gpu_indx),
                '--vis', vis,
            ]
            if run_identity['max_samples'] is not None:
                cmd += ['--max-samples', str(run_identity['max_samples'])]

            print(' '.join(cmd))
            returncode = subprocess.run(cmd).returncode

            if returncode != 0:
                print(f'LOI: {cls} thoat voi ma {returncode}')
                failures.append((cls, returncode))

                # Class dau tien hong gan nhu luon la moi truong hong (thieu
                # package, thieu checkpoint, sai duong dan dataset), khong phai
                # loi rieng cua class do. Dung ngay thay vi lap lai cung mot
                # loi cho moi class con lai.
                if len(failures) == 1:
                    print('Class dau tien da hong - nhieu kha nang la moi truong, '
                          'khong phai du lieu. Dung lai de ban doc loi o tren.')
                    break

    if failures:
        print(f'\n{len(failures)} class chay hong:')
        for cls, returncode in failures:
            print(f'  {cls}: ma thoat {returncode}')
        print('Cac class da chay xong van nam trong CSV; chay lai se tiep tuc '
              'tu cho do (resume theo tung class).')
        sys.exit(1)

    print('\nTat ca class chay xong.')
