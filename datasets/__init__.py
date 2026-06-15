import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from .dataset import SAADataset
from .ksdd2 import load_ksdd2, ksdd2_classes
from .mtd import load_mtd, mtd_classes
from .mvtec import load_mvtec, mvtec_classes
from .visa_challenge import load_visa_challenge, visa_challenge_classes
from .visa_public import load_visa_public, visa_public_classes

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'visa_challenge': load_visa_challenge,
    'visa_public': load_visa_public,
    'ksdd2': load_ksdd2,
    'mtd': load_mtd,

}

dataset_classes = {
    'mvtec': mvtec_classes,
    'visa_challenge': visa_challenge_classes,
    'visa_public': visa_public_classes,
    'ksdd2': ksdd2_classes,
    'mtd': mtd_classes,
}


def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x


def stratified_subset(dataset_inst, n):
    """Return indices for a stratified subsample keeping good/defect ratio."""
    labels = dataset_inst.labels
    good_idx = [i for i, l in enumerate(labels) if l == 0]
    defect_idx = [i for i, l in enumerate(labels) if l == 1]

    ratio = len(good_idx) / len(labels)
    n_good = max(1, round(n * ratio))
    n_defect = max(1, n - n_good)

    import random
    rng = random.Random(42)
    sampled_good = rng.sample(good_idx, min(n_good, len(good_idx)))
    sampled_defect = rng.sample(defect_idx, min(n_defect, len(defect_idx)))

    return sorted(sampled_good + sampled_defect)


def get_dataloader_from_args(phase, **kwargs):
    dataset_inst = SAADataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        experiment_indx=kwargs['experiment_indx']
    )

    max_samples = kwargs.get('max_samples')
    if phase == 'test' and max_samples is not None:
        from torch.utils.data import Subset
        indices = stratified_subset(dataset_inst, max_samples)
        dataset_inst = Subset(dataset_inst, indices)

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                 num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)

    debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    logger.info(debug_str)

    return data_loader, dataset_inst
