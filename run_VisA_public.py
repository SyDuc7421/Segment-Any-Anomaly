import json
import os
from datasets import dataset_classes
from multiprocessing import Pool

from utils.csv_utils import check_run_identity, completed_classes

if __name__ == '__main__':

    pool = Pool(processes=1)

    dataset_list = ['visa_public']
    gpu_indx = 0
    root_dir = './result'

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

    # Danh tinh cua lan chay nay - dung CHUNG mot dict de dung lenh command
    # line va de quyet dinh skip, tranh viec hai noi lech nhau. sam_variant
    # va saliency_backbone chua co CLI env o day (runner nay chi chay
    # baseline), nen khoa cung theo default cua eval_SAA.py.
    run_identity = {
        'max_samples': int(max_samples) if max_samples else None,
        'cal_pro': cal_pro.lower() in ('yes', 'true', 't', '1'),
        'sam_variant': 'vit_h',
        'saliency_backbone': 'wide_resnet50',
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
            # save_metric them tien to dataset cho moi dataset khac mvtec
            meta_key = f'{dataset}-{cls}'

            if meta_key in done:
                ok, reason = check_run_identity(meta, meta_key, run_identity)
                if ok:
                    print(f'skip {cls}: da co ket qua trong {csv_path}')
                    continue
                print(f'CANH BAO: {cls} co ket qua trong CSV nhung KHONG dang '
                      f'tin ({reason}) - chay lai va ghi de.')

            sh_method = (
                f'python eval_SAA.py '
                f'--dataset {dataset} '
                f'--class-name {cls} '
                f'--batch-size 1 '
                f'--root-dir {root_dir} '
                f'--cal-pro {run_identity["cal_pro"]} '
                f'--sam-variant {run_identity["sam_variant"]} '
                f'--saliency-backbone {run_identity["saliency_backbone"]} '
                f'--gpu-id {gpu_indx} '
            )
            if run_identity['max_samples'] is not None:
                sh_method += f'--max-samples {run_identity["max_samples"]} '

            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
