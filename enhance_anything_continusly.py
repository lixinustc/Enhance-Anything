import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from sam_segment import predict_masks_with_sam
# from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
import os.path as osp
import logging
import torch
import numpy as np
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument('--opt', type=str, required=True, help='Path to options YMAL file.')



if __name__ == "__main__":
    """Example usage:
    python enhance_anything_continuously.py \
        --input_img ./test_img/ \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama \
        --opt ./options/test/modulation_CResMD.yml
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = load_img_to_array(args.input_img)

    masks, _, _ = predict_masks_with_sam(
        img,
        [args.point_coords],
        args.point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )
    # print('======',img.max(), img.min())
    masks_ori = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks_ori]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.output_dir+'_mask')/img_stem
    mask_dir.mkdir(parents=True, exist_ok=True)
    out_fore_dir = Path(args.output_dir+'_fore')/img_stem
    out_fore_dir.mkdir(parents=True, exist_ok=True)
    # out_dir = Path(args.output_dir+'_fore')/img_stem
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = mask_dir / f"image_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"
        background_p = out_dir / f"background_image_{idx}.png"
        foreground_p = out_fore_dir / f"image_{idx}.png"

        # save the mask
        save_array_to_img(masks_ori[idx], mask_p)
        mask_ = mask[:, :, np.newaxis]
        background_img = img * (1-mask_/255.)
        front_img = img * (mask_/255.)
        save_array_to_img(background_img, background_p)
        save_array_to_img(front_img, foreground_p)
        # image_mask 
        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [args.point_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    ### enhance & restore anything

    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    print(opt)
    opt['datasets']['test_1']['dataroot_LQ'] = out_fore_dir
    opt['datasets']['test_1']['dataroot_GT'] = mask_dir
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        print('=====dataset_opt', dataset_opt)
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        # logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    stride = opt['modulation_stride'] if opt['modulation_stride'] is not None else 0.1
    cond = opt['cond_init']
    mod_dim = opt['modulation_dim']
    start_point = cond[mod_dim]

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        # logger.info('\nModulating [{:s}]...'.format(test_set_name))
        dataset_dir = osp.join('./results_all', test_set_name)
        util.mkdir(dataset_dir)
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True

        for coef in np.arange(start_point, 1.01, stride):
            print("setting coef to {:.2f}".format(coef))
            # logger.info('setting coef to {:.2f}'.format(coef))
            # load the test data
            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []
            test_results['psnr_y'] = []
            test_results['ssim_y'] = []

            for data in test_loader:
                cond[mod_dim] = coef
                data['cond'] = torch.Tensor(cond).view(1, -1)
                model.feed_data(data, need_GT=need_GT, need_cond=True)
                img_path = data['LQ_path'][0]
                img_name = osp.splitext(osp.basename(img_path))[0]
                img_dir = osp.join(dataset_dir, img_name)
                util.mkdir(img_dir)

                model.test()

                visuals = model.get_current_visuals(need_GT=need_GT)

                sr_img = util.tensor2img(visuals['rlt'])  # uint8
                mask_sr = util.tensor2img(visuals['GT'])
                mask_sr = mask_sr[:, :, np.newaxis]
                # sr_img = sr_img * (1-mask_sr/255.) + img *mask_sr/255.
                sr_img = sr_img[:, :, ::-1]* (mask_sr/255.) + img[:, :, ::-1]*(1-mask_sr/255.)
                # sr_img = sr_img[:, :, ::-1]
                # save images
                img_part_name = '_coef_{:.2f}.png'.format(coef)
                suffix = opt['suffix']
                if suffix:
                    save_img_path = osp.join(img_dir, img_name + suffix + img_part_name)
                else:
                    save_img_path = osp.join(img_dir, img_name + img_part_name)
                util.save_img(sr_img, save_img_path)
 