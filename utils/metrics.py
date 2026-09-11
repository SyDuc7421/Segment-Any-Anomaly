import numpy as np
from skimage import measure
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc

def calculate_max_f1(gt, scores):
    precision, recall, thresholds = precision_recall_curve(gt, scores)
    a = 2 * precision * recall
    b = precision + recall
    f1s = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    index = np.argmax(f1s)
    max_f1 = f1s[index]
    threshold = thresholds[index]
    return max_f1, threshold

def metric_cal(scores, gt_list, gt_mask_list, cal_pro=False):
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list, dtype=int)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    # print('INFO: image ROCAUC: %.3f' % (img_roc_auc))

    img_f1, img_threshold = calculate_max_f1(gt_list, img_scores)

    gt_mask = np.asarray(gt_mask_list, dtype=int)
    pxl_f1, pxl_threshold = calculate_max_f1(gt_mask.flatten(), scores.flatten())

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    ##############

    ap_im = average_precision_score(gt_list, img_scores)
    ap_px = average_precision_score(gt_mask.flatten(), scores.flatten())


    # calculate max-f1 region
    if cal_pro:
        pro_auc_score = cal_pro_metric(gt_mask_list, scores, fpr_thresh=0.3)
        max_f1_region = calculate_max_f1_region(gt_mask_list, scores)
        max_f1_region_fixed = calculate_max_f1_region_fixed(gt_mask_list, scores)
    else:
        pro_auc_score = 0
        max_f1_region = 0
        max_f1_region_fixed = 0

    result_dict = {
        'i_roc': img_roc_auc * 100,
        'p_roc': per_pixel_rocauc * 100,
        'i_ap': ap_im * 100,
        'p_ap': ap_px * 100,
        'i_f1': img_f1 * 100,
        # 'i_thresh': img_threshold,
        'p_f1': pxl_f1 * 100,
        # 'p_thresh': pxl_threshold,
        # Ban goc cua SAA+, giu lai CHI de so ngang voi bang da cong bo.
        # No co the vuot 100 (vd class 'wood' ra 122.57) vi recall dung
        # sai mau so - xem docstring cua calculate_max_f1_region_fixed.
        'r_f1': max_f1_region * 100,
        # Cai dung dinh nghia paper phat bieu. Day moi la so dung.
        'r_f1_fixed': max_f1_region_fixed * 100,
        'p_pro': pro_auc_score * 100,
    }

    return result_dict


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score

def calculate_max_f1_region(labeled_imgs, score_imgs, pro_thresh=0.6, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    # labeled_imgs[labeled_imgs <= 0.1] = 0
    # labeled_imgs[labeled_imgs > 0.1] = 1
    labeled_imgs = labeled_imgs.astype(bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    f1_list = []
    recall_list = []
    precision_list = []

    binary_score_maps = np.zeros_like(score_imgs, dtype=bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap

        predict_region_number = 0
        gt_region_number = 0

        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)

            score_map = measure.label(binary_score_maps[i], connectivity=2)
            score_props = measure.regionprops(score_map)

            predict_region_number += len(score_props)
            gt_region_number += len(props)

            # if len(score_props) == 0 or len(props) == 0:
            #     pro.append(0)
            #     continue

            for score_prop in score_props:
                x_min_0, y_min_0, x_max_0, y_max_0 = score_prop.bbox
                cur_pros = [0]
                for prop in props:
                    x_min_1, y_min_1, x_max_1, y_max_1 = prop.bbox

                    x_min = min(x_min_0, x_min_1)
                    y_min = min(y_min_0, y_min_1)
                    x_max = max(x_max_0, x_max_1)
                    y_max = max(y_max_0, y_max_1)

                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    cropped_gt_label = labeled_imgs[i][x_min:x_max, y_min:y_max]

                    # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = prop.filled_image  # corrected!
                    intersection = np.logical_and(cropped_pred_label, cropped_gt_label).astype(np.float32).sum()
                    union = np.logical_or(cropped_pred_label, cropped_gt_label).astype(np.float32).sum()
                    cur_pros.append(intersection / union)

                pro.append(max(cur_pros))

        pro = np.array(pro)

        if gt_region_number == 0 or predict_region_number == 0:
            print(f'gt_number: {gt_region_number}, pred_number: {predict_region_number}')
            recall = 0
            precision = 0
            f1 = 0
        else:
            recall = np.array(pro >= pro_thresh).astype(np.float32).sum() / gt_region_number
            precision = np.array(pro >= pro_thresh).astype(np.float32).sum() / predict_region_number

            if recall == 0 or precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)


        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)

    # as array
    f1_list = np.array(f1_list)
    max_f1 = f1_list.max()
    cor_recall = recall_list[f1_list.argmax()]
    cor_precision = precision_list[f1_list.argmax()]
    print(f'cor recall: {cor_recall}, cor precision: {cor_precision}')
    return max_f1


def _region_pair_ious(gt_label, gt_areas, n_gt, pred_label, n_pred):
    """IoU cua moi cap (vung du doan, vung GT) co giao nhau.

    Tinh tren mat na cua tung vung, khong phai tren cua so bbox chung nhu
    calculate_max_f1_region: cat ca ban do nhi phan theo bbox hop khien moi
    vung khac roi vao cua so do deu gop vao giao va hop, thoi phong IoU.

    Returns:
        List cac tuple (iou, chi_so_vung_du_doan, chi_so_vung_gt).
    """
    pred_areas = np.bincount(pred_label.ravel(), minlength=n_pred + 1)

    overlap = (pred_label > 0) & (gt_label > 0)
    if not overlap.any():
        return []

    keys = pred_label[overlap].astype(np.int64) * (n_gt + 1) + gt_label[overlap]
    counts = np.bincount(keys)

    pairs = []
    for key in np.nonzero(counts)[0]:
        p, g = divmod(int(key), n_gt + 1)
        intersection = int(counts[key])
        union = int(pred_areas[p]) + int(gt_areas[g]) - intersection
        if union > 0:
            pairs.append((intersection / union, p, g))

    return pairs


def _count_matched_pairs(pairs, pro_thresh):
    """Ghep mot-mot tham lam theo IoU giam dan, tra ve so cap khop duoc.

    Mot-mot la cho quyet dinh: no dam bao TP <= min(so vung du doan, so vung
    GT), nen precision va recall deu <= 1 va F1 khong the vuot 1.
    """
    matched_pred = set()
    matched_gt = set()
    true_positives = 0

    for iou, p, g in sorted(pairs, reverse=True):
        if iou < pro_thresh:
            break
        if p in matched_pred or g in matched_gt:
            continue
        matched_pred.add(p)
        matched_gt.add(g)
        true_positives += 1

    return true_positives


def calculate_max_f1_region_fixed(labeled_imgs, score_imgs, pro_thresh=0.6, max_steps=200):
    """max-F1-region theo dung dinh nghia phat bieu trong paper SAA+.

    Paper (docs/SAA+.md dong 332-337): "we compute the F1-score for
    region-wise segmentation at the optimal threshold, considering a
    prediction positive if the overlapping value exceeds 0.6".

    Khac calculate_max_f1_region o hai cho, va ca hai deu can thiet de F1
    khong vuot 1:

    1. TP la MOT con so, lay tu phep ghep mot-mot. Ban cu dem so vung DU DOAN
       khop duoc roi dung chinh con so do lam tu so cho ca precision lan
       recall - nen khi nhieu manh du doan cung trum mot vung GT thi
       recall = hits / so_vung_gt vuot 1. Vi du toi thieu: mot vung GT, du
       doan bi khe 1 pixel tach doi, ban cu tra ve 1.333.
    2. IoU tinh tren mat na tung vung thay vi tren cua so bbox chung cua ca
       ban do nhi phan.

    Giu nguyen pro_thresh=0.6 va max_steps=200 de so sanh duoc voi ban cu.
    """
    labeled_imgs = np.array(labeled_imgs).astype(bool)
    score_imgs = np.array(score_imgs)

    # Vung GT khong doi qua cac buoc threshold - gan nhan mot lan.
    gt_labels = []
    gt_counts = []
    gt_areas = []
    for img in labeled_imgs:
        label_map = measure.label(img, connectivity=2)
        n_gt = int(label_map.max())
        gt_labels.append(label_map)
        gt_counts.append(n_gt)
        gt_areas.append(np.bincount(label_map.ravel(), minlength=n_gt + 1))

    total_gt = sum(gt_counts)
    if total_gt == 0:
        return 0.

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    max_f1 = 0.
    for step in range(max_steps):
        thred = max_th - step * delta
        binary_score_maps = score_imgs > thred

        true_positives = 0
        total_pred = 0

        for i in range(len(binary_score_maps)):
            pred_label = measure.label(binary_score_maps[i], connectivity=2)
            n_pred = int(pred_label.max())
            total_pred += n_pred

            if n_pred == 0 or gt_counts[i] == 0:
                continue

            pairs = _region_pair_ious(
                gt_labels[i], gt_areas[i], gt_counts[i], pred_label, n_pred
            )
            true_positives += _count_matched_pairs(pairs, pro_thresh)

        if total_pred == 0 or true_positives == 0:
            continue

        precision = true_positives / total_pred
        recall = true_positives / total_gt
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > max_f1:
            max_f1 = f1

    return max_f1
