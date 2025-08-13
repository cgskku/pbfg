import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from basicsr.data.flare7k_dataset import Flare_Image_Loader, RandomGammaCorrection
from basicsr.archs.uformer_arch import Uformer
import argparse
from basicsr.archs.unet_arch import U_Net
from basicsr.utils.flare_util import blend_light_source, get_args_from_json, save_args_to_json, mkdir, \
    predict_flare_from_6_channel, predict_flare_from_3_channel, blend_light_source_zj
from torch.distributions import Normal
import torchvision.transforms as transforms
from skimage.exposure import match_histograms
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, default=None)
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--model_type', type=str, default='Uformer')
parser.add_argument('--model_path', type=str, default='checkpoint/flare7kpp/net_g_last.pth')
parser.add_argument('--output_ch', type=int, default=6)
parser.add_argument('--flare7kpp', action='store_const', const=True,
                    default=False)  # use flare7kpp's inference method and output the light source directly.

args = parser.parse_args()
model_type = args.model_type
images_path = os.path.join(args.input, "*.*")
result_path = args.output
pretrain_dir = args.model_path
output_ch = args.output_ch


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def load_params(model_path):
    full_model = torch.load(model_path)
    if 'params_ema' in full_model:
        return full_model['params_ema']
    elif 'params' in full_model:
        return full_model['params']
    else:
        return full_model


def demo(images_path, output_path, model_type, output_ch, pretrain_dir, flare7kpp_flag, gt_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    test_path = glob.glob(images_path)
    gt_path_files = glob.glob(os.path.join(gt_path, "*.*"))

    result_path = output_path
    torch.cuda.empty_cache()

    if model_type == 'Uformer':
        model = Uformer(img_size=512, img_ch=3, output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type == 'U_Net' or model_type == 'U-Net':
        model = U_Net(img_ch=3, output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    else:
        assert False, "This model is not supported!!"

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((512, 512))

    print("GT Path:", gt_path)

    for i, (image_path, gt_image_path) in tqdm(enumerate(zip(test_path, gt_path_files))):
        if not flare7kpp_flag:
            mkdir(os.path.join(result_path, "deflare/"))
            deflare_path = os.path.join(result_path, "deflare/", f"{str(i).zfill(5)}_deflare.png")

        mkdir(os.path.join(result_path, "flare/"))
        mkdir(os.path.join(result_path, "input/"))
        mkdir(os.path.join(result_path, "blend/"))

        flare_path = os.path.join(result_path, "flare/", f"{str(i).zfill(5)}_flare.png")
        merge_path = os.path.join(result_path, "input/", f"{str(i).zfill(5)}_input.png")
        blend_path = os.path.join(result_path, "blend/", f"{str(i).zfill(5)}_blend.png")

        img_gt = Image.open(gt_image_path).convert("RGB")
        img_gt = resize(to_tensor(img_gt))
        img_gt = img_gt.cuda().unsqueeze(0)

        merge_img = Image.open(image_path).convert("RGB")
        merge_img = resize(to_tensor(merge_img))
        merge_img = merge_img.cuda().unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output_img = model(merge_img)
            gamma = torch.Tensor([2.2])
            if output_ch == 6:
                deflare_img, flare_img_predicted, merge_img_predicted = predict_flare_from_6_channel(output_img, gamma)
            elif output_ch == 3:
                flare_mask = torch.zeros_like(merge_img)
                deflare_img, flare_img_predicted = predict_flare_from_3_channel(
                    output_img, flare_mask, output_img, merge_img, merge_img, gamma
                )
            else:
                assert False, "This output_ch is not supported!!"

            torchvision.utils.save_image(merge_img, merge_path)
            torchvision.utils.save_image(flare_img_predicted, flare_path)


            if flare7kpp_flag:
                img_gt_cpu = img_gt.cpu().numpy()
                deflare_img_cpu = deflare_img.cpu().numpy()
                blend_img_matched = match_histograms(deflare_img_cpu, img_gt_cpu, channel_axis=0)
                blend_img_tensor = torch.from_numpy(blend_img_matched).float()
                blend_img_tensor = blend_img_tensor.to(deflare_img.device)
                torchvision.utils.save_image(blend_img_tensor, blend_path)
                # torchvision.utils.save_image(deflare_img, blend_path)
            else:
                blend_img = blend_light_source(merge_img, deflare_img, 0.97)
                torchvision.utils.save_image(deflare_img, deflare_path)
                torchvision.utils.save_image(blend_img, blend_path)

gt_path = args.gt
demo(images_path, result_path, model_type, output_ch, pretrain_dir, args.flare7kpp,gt_path)