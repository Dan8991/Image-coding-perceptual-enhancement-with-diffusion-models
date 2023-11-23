"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from improved_diffusion.train_util import TrainLoop
import wandb


def main():
    args = create_argparser().parse_args()

    wandb.init(project="diffusion-compression", config=args, mode=args.wandb_mode, name=args.run_name)
    if args.use_distributed:
        from improved_diffusion import dist_util
        dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    local_device = "cuda" if th.cuda.is_available() else "cpu"
    device = local_device if not args.use_distributed else dist_util.dev()
    model.to(device)
    diffusion.encoder.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    model_total_parameters = sum(p.numel() for p in model.parameters())
    enc_total_parameters = sum(p.numel() for p in diffusion.encoder.parameters())
    print("Number of parameters in the decoder: ", model_total_parameters)
    print("Number of parameters in the encoder: ", enc_total_parameters)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        validation_dir=args.validation_dir,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_distributed=args.use_distributed,
        use_ddim=args.use_ddim,
        clip_denoised=args.clip_denoised,
        max_steps=args.max_steps,
        final_RD=args.final_RD,
        p_guidance=args.p_guidance,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        validation_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0e1,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_distributed=False,
        wandb_mode="online",
        use_ddim=False,
        clip_denoised=True,
        channel_mult="1,2,3,4",
        encoder_qp=1,
        pretrained_enc=True,
        RD_lambda = 1e-4,
        rho=0,
        codec = "FP",
        max_steps=1e5,
        final_RD=0.0128,
        conditioned_decoder=False,
        p_guidance=0.0,
        run_name=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
