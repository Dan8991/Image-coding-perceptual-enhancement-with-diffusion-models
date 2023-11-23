"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from pathlib import Path

from improved_diffusion import logger
from improved_diffusion.image_datasets import load_data, _list_image_files_recursively
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_gaussian_diffusion,
)
from improved_diffusion.respace import space_timesteps
from torchvision.utils import make_grid
from torchvision.transforms import Resize, CenterCrop
from compressai.zoo import bmshj2018_factorized
import matplotlib.pyplot as plt
from glob import glob
from torchvision.utils import save_image
import json
from torchvision.io import read_image
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import multiscale_structural_similarity_index_measure, peak_signal_noise_ratio
from tqdm import tqdm
from compute_metrics import read_images

def get_rate(strings, batch_shape, i):
    if len(strings) == 1:
        rate_z = 0
    else:
        rate_z = len(strings[1][i]) * 8 / np.product(batch_shape[2:]) 
    rate_y = len(strings[0][i]) * 8 / np.product(batch_shape[2:]) 
    return rate_z + rate_y 

def evaluate_fp(batch, encoder_qp):
    rates = []
    psnrs = []
    batch = (batch + 1) / 2
    rec = None
    for qp in range(1, 9):
        model = bmshj2018_factorized(quality=qp, pretrained=True)
        latent = model.compress(batch)
        x_hat = model.decompress(latent["strings"], latent["shape"])["x_hat"]
        if qp == encoder_qp:
            rec = x_hat
        rate = [
            len(latent["strings"][0][i]) * 8 / np.product(batch.shape[2:]) 
            for i in range(len(batch))
        ]
        mse = th.nn.MSELoss(reduction="none")(x_hat, batch)
        mse = mse.reshape(len(batch), -1).mean(axis=1)
        psnr = 10 * th.log10(1 / mse).detach().cpu().numpy()
        rates.append(rate)
        psnrs.append(psnr)
    rates = np.array(rates)
    psnrs = np.array(psnrs)
    return rates, psnrs, rec

def main():
    args = create_argparser().parse_args()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
       th.load(args.model_path)
    )
    diffusion.encoder.update()
    diffusion.encoder.load_state_dict(th.load(args.model_path[:-3]+"_encoder.pt"))
    diffusion.encoder.update(force="True")

    device = "cuda" if th.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    diffusion.encoder.to(device)
    diffusion.encoder.eval()

    logger.log("sampling...")
    ind = 0
    tmp_folder = os.path.join(
        "tmp",
        f"diff_{args.codec}_{args.encoder_qp if args.pretrained_enc else args.RD_lambda}"
    )
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)
    timestep_respacings = [2, 5] + list(range(10, 30, 5))

    inception = InceptionScore(splits=2)
    kid = KernelInceptionDistance(subsets=1, subset_size=1)
    fid = FrechetInceptionDistance()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

    psnrs = []
    kids = []
    fids = []
    lpipss = []
    ssims = []
    images = read_images("datasets/kodak").float()
    images = images / 127.5 - 1
    images = images.to(device)
    images = images[:6]

    with th.no_grad():
        for image_id, image in enumerate(images):
            psnrs.append([])
            kids.append([])
            fids.append([])
            lpipss.append([])
            ssims.append([])
            image = image.unsqueeze(0)
            for timestep_respacing in tqdm(timestep_respacings):
                diffusion  = create_gaussian_diffusion(
                    steps=args.diffusion_steps,
                    learn_sigma=True,
                    noise_schedule=args.noise_schedule,
                    use_kl=args.use_kl,
                    predict_xstart=args.predict_xstart,
                    rescale_timesteps=args.rescale_timesteps,
                    rescale_learned_sigmas=args.rescale_learned_sigmas,
                    timestep_respacing=[timestep_respacing],
                    RD_lambda=args.RD_lambda,
                    rho=args.rho,
                    sigma_small=args.sigma_small,
                    num_layers=3,
                    encoder_qp=1,
                    pretrained_enc=False,
                    codec=args.codec
                )
                diffusion.encoder.update()
                diffusion.encoder.load_state_dict(th.load(args.model_path[:-3]+"_encoder.pt"))
                diffusion.encoder.update(force="True")
                diffusion.encoder.to(device)
                diffusion.encoder.eval()

                model_kwargs = {}
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                compressed, _ = diffusion.encoder(image)
                strings = diffusion.encoder.compress(
                    image
                )["strings"] 
                sample = sample_fn(
                    model,
                    compressed,
                    image.shape,
                    clip_denoised=args.clip_denoised,
                    noise=th.zeros_like(image),
                    model_kwargs=model_kwargs,
                )

                image_u =  ((image + 1) * 127.5).to(th.uint8)
                sample_u =  ((sample + 1) * 127.5).to(th.uint8)

                kid.cuda()
                kid.update(image_u.cuda(), real=True)
                kid.update(sample_u.cuda(), real=False)
                kid_mean, _ = kid.compute()
                kids[image_id].append(kid_mean.cpu())
                kid.cpu()

                fids[image_id].append(1)
                # fid.cuda()
                # fid.update(image_u, real=True)
                # fid.update(sample_u.cuda(), real=False)
                # fid_mean = fid.compute().cpu()
                # fids.append(fid_mean)
                # fid.cpu()

                lpips.cuda()
                norm_orig = image.cuda().float() 
                norm_images = sample.cuda().float() 
                lpipss[image_id].append(lpips(norm_orig, norm_images).detach().cpu())
                lpips.cpu()

                ssims[image_id].append(
                    multiscale_structural_similarity_index_measure(
                        sample_u.cuda().float(), image_u.float()
                    ).cpu()
                )

                psnrs[image_id].append(
                    peak_signal_noise_ratio(sample_u.cuda().float(), image_u.float()).cpu()
                )
                diffusion.encoder.cpu()
    
    fig, axs = plt.subplots(2, 3)
    for image_id in range(len(fids)):
        axs[0, 0].plot(timestep_respacings, fids[image_id])
        axs[0, 1].plot(timestep_respacings, kids[image_id])
        axs[0, 2].plot(timestep_respacings, ssims[image_id])
        axs[1, 0].plot(timestep_respacings, lpipss[image_id])
        axs[1, 1].plot(timestep_respacings, psnrs[image_id])
    axs[0, 0].set_title("FID")
    axs[0, 1].set_title("KID")
    axs[0, 2].set_title("MSSIM")
    axs[1, 0].set_title("LPIPS")
    axs[1, 1].set_title("PSNR")
    plt.show()

def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        use_distributed=False,
        channel_mult="1,2,3,4",
        RD_lambda = 1e-1,
        encoder_qp=1,
        codec = "FP"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
