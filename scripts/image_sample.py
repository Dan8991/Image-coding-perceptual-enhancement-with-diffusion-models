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
)
from torchvision.utils import make_grid
from torchvision.transforms import Resize, CenterCrop
from compressai.zoo import bmshj2018_factorized
import matplotlib.pyplot as plt
from glob import glob
from torchvision.utils import save_image
import json

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

    if args.use_distributed:
        from improved_diffusion import dist_util 
        dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.use_distributed:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
    else:
        model.load_state_dict(
           th.load(args.model_path)
        )
        diffusion.encoder.update()
        diffusion.encoder.load_state_dict(th.load(args.model_path[:-3]+"_encoder.pt"))
        diffusion.encoder.update(force="True")

    local_device = "cuda" if th.cuda.is_available() else "cpu"
    device = local_device if not args.use_distributed else dist_util.dev()
    model.to(device)
    model.eval()
    diffusion.encoder.to(device)
    diffusion.encoder.eval()

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        sampling=True,
        deterministic=True
    )

    logger.log("sampling...")
    all_images = []
    all_labels = []
    rates = [] 
    psnrs = []
    diff_sizes = []
    diff_psnrs = []
    images_path = _list_image_files_recursively(args.data_dir)
    ind = 0
    tmp_folder = os.path.join(
        "tmp",
        f"diff_{args.codec}_{args.encoder_qp if args.pretrained_enc else args.RD_lambda}"
    )
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)
    with th.no_grad():
        while len(all_images) * args.batch_size < args.num_samples:
            batch, _ = next(data)
            batch = (batch + 1) / 2
            image_name = os.path.basename(images_path[ind])
            # batch = Resize((batch.shape[-2]//2, batch.shape[-1]//2))(batch)
            # batch = CenterCrop((512, 512))(batch)
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=local_device
                )
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            batch = batch.to(device)
            compressed, _ = diffusion.encoder(batch)
            strings = diffusion.encoder.compress(
                batch
            )["strings"] 
            diff_sizes += [
                get_rate(strings, batch.shape, i) for i in range(len(strings[0]))
            ]
            sample = sample_fn(
                model,
                compressed,
                batch.shape,#(args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                save_intermediate=True
            )
            mse = th.nn.MSELoss(reduction="none")(sample.float(), (batch + 1)*127.5)
            mse = mse.reshape((len(mse), -1)).mean(axis=1)
            psnr = 10*th.log10(255**2/mse).detach().cpu().numpy()
            diff_psnrs += list(psnr)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            world_size = 1 if not args.use_distributed else dist.get_world_size()
            if args.use_distributed:
                gathered_samples = [th.zeros_like(sample) for _ in range(world_size)]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            else:
                gathered_samples=[sample]
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")
            save_image(
                sample[0].permute(2, 0, 1).float() / 255,
                os.path.join(tmp_folder, image_name)
            )
            quit()
            image_grid = make_grid(
                sample.permute(0, 3, 1, 2).cpu(),
                nrow=args.batch_size
            ).permute(1, 2, 0).numpy()
            batch = ((batch + 1)*127.5).to(th.uint8)
            image_grid_orig = make_grid(
                batch,
                nrow=args.batch_size
            ).permute(1, 2, 0)
            ind += 1


    with open(os.path.join(tmp_folder, "rd.json"), "w") as f:
        json.dump({"rates": rates}, f, indent=2)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if not args.use_distributed:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    if args.use_distributed:
        dist.barrier()
    logger.log("sampling complete")


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
