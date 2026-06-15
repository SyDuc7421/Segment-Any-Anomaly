import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)

    dataset_list = ['visa_public']
    gpu_indx = 0
    max_samples = int(os.environ.get('MAX_SAMPLES', 42))   # 42 × 12 classes ≈ 504 imgs

    for dataset in dataset_list:
        classes = dataset_classes[dataset]
        for cls in classes[:]:
            sh_method = (
                f'python eval_SAA.py '
                f'--dataset {dataset} '
                f'--class-name {cls} '
                f'--batch-size 1 '
                f'--root-dir ./result '
                f'--cal-pro False '
                f'--gpu-id {gpu_indx} '
                f'--max-samples {max_samples} '
            )
            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
