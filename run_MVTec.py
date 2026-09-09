import os
from datasets import dataset_classes
from multiprocessing import Pool

from utils.csv_utils import completed_classes

if __name__ == '__main__':

    pool = Pool(processes=1)

    dataset_list = ['mvtec']
    gpu_indx = 0
    root_dir = './result'

    # Khong dat mac dinh. Spec muc 6.1: moi con so vao luan van phai chay full
    # test set, vi normalize() phu thuoc vao tap anh cua lan chay do.
    max_samples = os.environ.get('MAX_SAMPLES')

    for dataset in dataset_list:
        csv_path = os.path.join(root_dir, 'csv', f'{dataset}-indx-0.csv')
        done = completed_classes(csv_path)

        classes = dataset_classes[dataset]
        for cls in classes[:]:
            if cls in done:
                print(f'skip {cls}: da co ket qua trong {csv_path}')
                continue

            sh_method = (
                f'python eval_SAA.py '
                f'--dataset {dataset} '
                f'--class-name {cls} '
                f'--batch-size 1 '
                f'--root-dir {root_dir} '
                f'--cal-pro False '
                f'--gpu-id {gpu_indx} '
            )
            if max_samples is not None:
                sh_method += f'--max-samples {max_samples} '

            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
