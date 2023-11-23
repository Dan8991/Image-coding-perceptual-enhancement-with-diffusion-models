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
    if args.json_file is not None:
        with open(args.json_file) as f:
            json_args = json.load(f)
        for k, v in json_args.items():
            setattr(args, k, v)

    if args.use_distributed:
        from improved_diffusion import dist_util 
        dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    models_paths = args.model_path

    for model_folder in set(glob(os.path.join(models_paths, "*"))) - set(glob(os.path.join(models_paths, "*.json"))):
        checkpoints = glob(os.path.join(model_folder, "ema_*.pt"))
        step_ids = [int(
            os.path.basename(checkpoint).split("_")[2].split(".")[0]
        ) for checkpoint in checkpoints]
        best_id = max(step_ids)
        
        model_path = os.path.join(model_folder, f"ema_0.9999_{best_id}.pt")
        if "qp" in model_path:
            # extract int after qp in model_path
            encoder_qp = int(model_folder.split("qp")[-1][-1])
            args.encoder_qp = encoder_qp
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        logger.log(f"sampling with model {model_path}...")

        model.load_state_dict(
           th.load(model_path)
        )
        diffusion.encoder.update()
        diffusion.encoder.load_state_dict(th.load(model_path[:-3]+"_encoder.pt"))
        diffusion.encoder.update(force="True")

        local_device = "cuda" if th.cuda.is_available() else "cpu"
        device = local_device if not args.use_distributed else dist_util.dev()
        model.to(device)
        model.eval()
        diffusion.encoder.to(device)
        diffusion.encoder.eval()

        data = load_data(
            data_dir=args.data_dir,
            batch_size=1,
            image_size=args.image_size,
            class_cond=args.class_cond,
            sampling=True,
            deterministic=True
        )

        all_images = []
        all_labels = []
        rates = [] 
        psnrs = []
        diff_sizes = []
        images_path = _list_image_files_recursively(args.data_dir)
        ind = 0
        path_split = model_folder.split(os.sep)
        family_name = path_split[1]
        model_name = path_split[2]
        tmp_folder = os.path.join(
            "tmp",
            family_name,
            model_name,
        )
        Path(tmp_folder).mkdir(parents=True, exist_ok=True)
        with th.no_grad():
            while len(all_images) * args.batch_size < args.num_samples:
                transposed = []
                batch = []
                image_names = []
                for i in range(args.batch_size):
                    temp_batch, _ = next(data)
                    if len(transposed) == 0:
                        transposed.append(False)
                        batch.append(temp_batch)
                    else:
                        if temp_batch.shape != batch[0].shape:
                            transposed.append(True)
                            batch.append(temp_batch.transpose(2, 3))
                        else:
                            transposed.append(False)
                            batch.append(temp_batch)
                    image_names.append(os.path.basename(images_path[ind]))
                    ind += 1
                batch = th.cat(batch, dim=0)
                model_kwargs = {}
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                batch = batch.to(device)
                if args.pretrained_enc:
                    batch =  (batch + 1) / 2
                compressed, _ = diffusion.encoder(batch)
                compressed = th.round(compressed)
                diffusion.encoder = diffusion.encoder.to("cpu")
                strings = diffusion.encoder.compress(
                    batch.to("cpu")
                )["strings"] 
                diffusion.encoder = diffusion.encoder.to(device)
                diff_sizes += [
                    get_rate(strings, batch.shape, i) for i in range(len(strings[0]))
                ]
                zero_noise = args.zero_noise
                noise = th.zeros_like(batch).to(device) if zero_noise else None
                sample = sample_fn(
                    model,
                    compressed,
                    batch.shape,
                    clip_denoised=args.clip_denoised,
                    noise=noise,
                    model_kwargs=model_kwargs,
                    save_intermediate=True,
                    stop_guidance=args.stop_guidance,
                )
                mse = th.nn.MSELoss(reduction="none")(sample.float(), (batch + 1)*127.5)
                if args.pretrained_enc:
                    sample = (sample * 255).clamp(0, 255).to(th.uint8)
                else:
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
                logger.log(f"created {len(all_images) * args.batch_size} samples")
                for image, image_name, transp in zip(sample, image_names, transposed):
                    if transp:
                        image = image.transpose(0, 1)
                    save_image(
                        image.permute(2, 0, 1).float().detach().cpu() / 255,
                        os.path.join(tmp_folder, image_name)
                    )

        with open(os.path.join(tmp_folder, "rd.json"), "w") as f:
            json.dump({"rates": diff_sizes}, f, indent=2)

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
        codec = "FP",
        json_file=None,
        zero_noise=False,
        stop_guidance=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
