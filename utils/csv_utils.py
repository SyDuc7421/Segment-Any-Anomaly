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
    """
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path, index_col=0)

    if metric_key not in df.columns:
        return set()

    return set(df.index[df[metric_key] > 0].astype(str))
