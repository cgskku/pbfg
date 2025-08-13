import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
from torchvision.transforms import ToTensor
import numpy as np
from glob import glob
import lpips

import warnings

warnings.filterwarnings("ignore")


def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
    return output_lpips.cpu().detach().numpy()[0, 0, 0, 0]


def compare_score(img1, img2, img_seg):
    mask_type_list = ['glare', 'streak', 'global']
    metric_dict = {'glare': 0, 'streak': 0, 'global': 0}

    for mask_type in mask_type_list:
        mask_area, img_mask = extract_mask(img_seg)[mask_type]
        if mask_area > 0:
            img_gt_masked = img1 * img_mask
            img_input_masked = img2 * img_mask
            input_mse = compare_mse(img_gt_masked, img_input_masked) / (255 * 255 * mask_area)
            input_psnr = 10 * np.log10((1.0 ** 2) / input_mse)
            metric_dict[mask_type] = round(input_psnr, 3)  # Round to 3 decimal places
        else:
            metric_dict.pop(mask_type)

    return metric_dict


def extract_mask(img_seg):
    mask_dict = {}
    streak_mask = (img_seg[:, :, 0] - img_seg[:, :, 1]) / 255
    glare_mask = img_seg[:, :, 1] / 255
    global_mask = (255 - img_seg[:, :, 2]) / 255

    mask_dict['glare'] = [np.sum(glare_mask) / (512 * 512), np.expand_dims(glare_mask, 2).repeat(3, axis=2)]
    mask_dict['streak'] = [np.sum(streak_mask) / (512 * 512), np.expand_dims(streak_mask, 2).repeat(3, axis=2)]
    mask_dict['global'] = [np.sum(global_mask) / (512 * 512), np.expand_dims(global_mask, 2).repeat(3, axis=2)]

    return mask_dict


def calculate_metrics(args):
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    gt_folder = args['gt'] + '/*'
    input_folder = args['input'] + '/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))

    if args['mask'] is not None:
        mask_folder = args['mask'] + '/*'
        mask_list = sorted(glob(mask_folder))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val = 0, 0, 0
    score_dict = {'glare': 0, 'streak': 0, 'global': 0, 'glare_num': 0, 'streak_num': 0, 'global_num': 0}

    for i in range(n):  # Removed tqdm from loop
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])

        ssim_val = compare_ssim(img_gt, img_input, multichannel=True)
        ssim += ssim_val

        psnr_val = compare_psnr(img_gt, img_input, data_range=255)
        psnr += psnr_val

        lpips_val_alex = compare_lpips(img_gt, img_input, loss_fn_alex)
        lpips_val += lpips_val_alex

        g_psnr, s_psnr = "N/A", "N/A"
        if args['mask'] is not None:
            img_seg = io.imread(mask_list[i])
            metric_dict = compare_score(img_gt, img_input, img_seg)

            if 'glare' in metric_dict:
                g_psnr = metric_dict['glare']
                score_dict['glare'] += g_psnr
                score_dict['glare_num'] += 1

            if 'streak' in metric_dict:
                s_psnr = metric_dict['streak']
                score_dict['streak'] += s_psnr
                score_dict['streak_num'] += 1

        print(
            f"Index: {i}, PSNR: {psnr_val:.3f}, SSIM: {ssim_val:.3f}, LPIPS: {lpips_val_alex:.4f}, G-PSNR: {g_psnr}, S-PSNR: {s_psnr}"
        )

    ssim /= n
    psnr /= n
    lpips_val /= n

    print(f"\nFinal Metrics:")
    print(f"PSNR: {psnr:.3f}, SSIM: {ssim:.3f}, LPIPS: {lpips_val:.4f}")

    if args['mask'] is not None:
        for key in ['glare', 'streak', 'global']:
            if score_dict[key + '_num'] == 0:
                print(f"Warning: No {key} masks found in dataset.")
            else:
                score_dict[key] /= score_dict[key + '_num']

        score_dict['score'] = (score_dict['glare'] + score_dict['global'] + score_dict['streak']) / 3
        print(
            f"Score: {score_dict['score']:.3f}, G-PSNR: {score_dict['glare']:.3f}, S-PSNR: {score_dict['streak']:.3f}, Global-PSNR: {score_dict['global']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    args = vars(parser.parse_args())
    calculate_metrics(args)