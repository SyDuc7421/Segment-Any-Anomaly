import pandas as pd
import os

def write_results(results:dict, cur_class, total_classes, csv_path):
    keys = list(results.keys())

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[class_name])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys:
        df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')

def save_metric(metrics, total_classes, class_name, dataset, csv_path):
    if dataset != 'mvtec':
        for indx in range(len(total_classes)):
            total_classes[indx] = f"{dataset}-{total_classes[indx]}"
        class_name = f"{dataset}-{class_name}"
    write_results(metrics, class_name, total_classes, csv_path)

def completed_classes(csv_path, metric_key='p_ap'):
    """Ten cac class da co ket qua that trong CSV.

    write_results khoi tao moi class bang 0.00 truoc khi co so, nen dieu kien
    "da xong" la metric_key > 0 chu khong phai "co dong trong file".

    Luu y ve default metric_key='p_ap': eval_SAA.py dung nhanh rieng cho
    dataset 'visa_challenge' xay dung result_dict KHONG co key 'p_ap' (xem
    eval_SAA.py, nhanh `if dataset in ['visa_challenge']`). Voi dataset do,
    ham nay se luon tra ve set rong (metric_key not in df.columns). Hien tai
    khong runner nao target 'visa_challenge' nen day la latent, khong phai bug.
    """
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path, index_col=0)

    if metric_key not in df.columns:
        return set()

    return set(df.index[df[metric_key] > 0].astype(str))


def check_run_identity(meta, meta_key, identity):
    """So sanh danh tinh cua lan chay hien tai voi entry da luu trong run_meta.json.

    So sanh gia tri --max-samples (upper bound cua invocation) chu khong
    phai n_images quan sat duoc: --max-samples la can tren va stratified
    subsampling co the cho ra it anh hon, nen so sanh n_images la mo ho con
    so sanh flag la chinh xac.

    Args:
        meta: dict da load tu run_meta.json (rong neu file khong ton tai).
        meta_key: khoa dung trong run_meta.json cho class nay, dang
            "{dataset}-{class_name}" (xem eval_SAA.py).
        identity: dict cac truong quyet dinh so lieu co so sanh duoc khong
            voi lan chay truoc - vi du max_samples, cal_pro, sam_variant,
            saliency_backbone.

    Returns:
        (ok, reason). ok=True va reason='' khi moi truong trong identity
        khop voi entry da luu. ok=False khi class chua tung duoc ghi
        metadata (CSV legacy hoac run_meta.json khong ton tai) hoac khi mot
        truong bat ky lech nhau - reason neu ro ly do de in canh bao.
    """
    if meta_key not in meta:
        return False, f'khong co entry trong run_meta.json cho {meta_key} (CSV legacy hoac thieu metadata)'

    entry = meta[meta_key]
    for field, expected in identity.items():
        actual = entry.get(field, '<missing>')
        if actual != expected:
            return False, f'{field} lech: run_meta={actual!r} lan chay hien tai={expected!r}'

    return True, ''
