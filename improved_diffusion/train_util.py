import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.transforms import Resize
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from improved_diffusion.image_datasets import load_data

from . import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .unet import Encoder
import wandb
from torchvision.utils import make_grid
from compressai.zoo import bmshj2018_factorized
import matplotlib.pyplot as plt

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        validation_dir,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=1e0,
        use_distributed=False,
        use_ddim=False,
        clip_denoised=True,
        max_steps=1e5,
        final_RD=0.0128,
        p_guidance=0
    ):
        if use_distributed:
            from . import dist_util
        else:
            dist_util = None
        self.model = model
        self.diffusion = diffusion
        self.data = data
        val_data = load_data(
            data_dir = validation_dir,
            batch_size = 1,
            image_size = 512,
            class_cond = False,
            sampling = False,
            deterministic = True,
            augmentation = False
        )
        self.val_data = [next(val_data)[0] for _ in range(16)]
        self.wandb_log_interval = 100000
        
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.use_distributed = use_distributed
        self.lr = lr
        self.p_guidance = p_guidance
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.clip_denoised = clip_denoised
        self.use_ddim = use_ddim
        self.final_RD = final_RD 

        self.step = 0
        self.resume_step = 0
        world_size = 1 if not use_distributed else dist.get_world_size()
        self.global_batch = self.batch_size * world_size

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self.max_steps = max_steps

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        opt_parameters = list(self.master_params)
        if not diffusion.pretrained_enc:
            opt_parameters += list(diffusion.encoder.parameters())
        self.opt = AdamW(
            opt_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available() or not use_distributed:
            self.use_ddp = True
            if not use_distributed:
                self.ddp_model= self.model
            else:
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            self.step += self.resume_step
            self.max_steps = self.max_steps + self.resume_step
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(
                    resume_checkpoint
                )
            )
            enc_state_dict = th.load(os.path.splitext(resume_checkpoint)[0] + "_encoder.pt")
            
            enc_state_dict.pop("entropy_bottleneck._offset")
            enc_state_dict.pop("entropy_bottleneck._quantized_cdf")
            enc_state_dict.pop("entropy_bottleneck._cdf_length")
            enc_state_dict.pop("gaussian_conditional._offset")
            enc_state_dict.pop("gaussian_conditional._quantized_cdf")
            enc_state_dict.pop("gaussian_conditional._cdf_length")
            enc_state_dict.pop("gaussian_conditional.scale_table")
            self.diffusion.encoder.load_state_dict(enc_state_dict, strict=False)
            self.diffusion.encoder.update()

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(
                ema_checkpoint
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint
            )
            self.opt.load_state_dict(state_dict)
            self.lr = self.opt.param_groups[0]["lr"]

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            self.step < self.max_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.wandb_log_interval == 0 and self.step > self.resume_step:
                self.run_validation()
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        print("Exiting Here, max_steps = {self.max_steps}, step: {self.step}")

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def evaluate_fp(self, batch):
        rates = []
        psnrs = []
        batch = (batch + 1) / 2
        for qp in range(1, 9):
            model = bmshj2018_factorized(quality=qp, pretrained=True)
            latent = model.compress(batch)
            x_hat = model.decompress(latent["strings"], latent["shape"])["x_hat"]
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
        return rates, psnrs

    def run_validation(self):
        '''
        Code taken and adapted from scripts/image_samples.py since I actually want to generate data
        to see how the training is going
        '''

        if self.use_distributed:
            from . import dist_util
        all_images = []
        all_labels = []
        local_device = "cuda" if th.cuda.is_available() else "cpu"
        device = local_device if not self.use_distributed else dist_util.dev()
        grid_size = 16
        rates, psnrs = [], []
        self.diffusion.encoder.eval()
        self.diffusion.encoder.update(force=True)
        self.model.eval()
        distortion_loss = th.nn.MSELoss(reduction="none")
        diff_sizes = []
        diff_psnrs = []
        diff_lpips = []
        batch = []
        with th.no_grad():
            while len(all_images) < grid_size:
                model_kwargs = {}
                num_classes = self.model.num_classes
                if num_classes is not None:
                    classes = th.randint(
                        low=0, high=num_classes, size=(self.microbatch,), device=device
                    )
                    model_kwargs["y"] = classes
                sample_fn = (
                    self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop
                )
                minibatch = self.val_data[len(all_images)]
                batch.append(minibatch)
                rate, psnr = self.evaluate_fp(minibatch)
                minibatch = minibatch.to(device)
                rates.append(rate)
                psnrs.append(psnr)
                if self.diffusion.pretrained_enc:
                    compress_input = (minibatch + 1) / 2 
                else:
                    compress_input = minibatch
                compressed, _ = self.diffusion.encoder(
                    compress_input 
                )
                strings = self.diffusion.encoder.compress(
                    compress_input
                )["strings"] 
                norm_const = 8 / np.product(minibatch.shape[2:]) 
                diff_sizes += [
                    len(string) * norm_const for string in strings[0]
                ]
                sample = sample_fn(
                    self.model,
                    compressed,
                    minibatch.shape,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                    save_intermediate=False
                )
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                mse = distortion_loss(sample.float(), (minibatch + 1)*127.5)
                mse = mse.reshape((len(mse), -1)).mean(axis=1)
                psnr = 10*th.log10(255**2/mse).detach().cpu().numpy()
                lpips = self.diffusion.lpips(
                    (sample.float() - 127.5) / 127.5,
                    minibatch
                )
                diff_psnrs.append(psnr)
                diff_lpips.append(lpips.detach().cpu().numpy())
                diff_sizes += [
                    get_rate(strings, minibatch.shape, i) for i in range(len(strings[0]))
                ]

                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                world_size = 1 if not self.use_distributed else dist.get_world_size()
                if self.use_distributed:
                    gathered_samples = [th.zeros_like(sample) for _ in range(world_size)]
                    dist.all_gather(gathered_samples, sample)  # gather not supported with nccl
                else:
                    gathered_samples=[sample]
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                if num_classes is not None:
                    gathered_labels = [
                        th.zeros_like(classes) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logger.log(f"created {len(all_images)} samples")


            rates = np.concatenate(rates, axis=1)
            psnrs = np.concatenate(psnrs, axis=1)
            batch = np.concatenate(batch, axis=0)
            self.diffusion.encoder.train()
            self.model.train()
            fig, axs = plt.subplots(4, 4)
            for i in range(4):
                for j in range(4):
                    axs[i, j].plot(rates[:, i + j*4], psnrs[:, i + j*4])
                    axs[i, j].scatter(diff_sizes[i + j*4], diff_psnrs[i + j*4])

            arr = th.tensor(np.concatenate(all_images, axis=0))
            batch = th.tensor(batch)
            mse = th.nn.MSELoss()((batch + 1) * 127.5, arr.permute(0, 3, 1, 2).float()) 
            mse = mse / 255**2
            if num_classes is not None:
                label_arr = np.concatenate(all_labels, axis=0)
            image_grid = make_grid(arr.permute(0, 3, 1, 2), nrow=4).permute(1, 2, 0).numpy()
            image_grid_orig = make_grid(batch, nrow=4).permute(1, 2, 0).numpy()
            plt.tight_layout()
            log_dict = {
                "val_images": wandb.Image(image_grid),
                "val_mse": mse,
                "val_psnr": np.mean(diff_psnrs),
                "val_lpips": np.mean(diff_lpips),
                "val_rate": np.mean(diff_sizes),
                "RD": wandb.Image(plt)
            }
            if self.step == self.wandb_log_interval:
                log_dict["real_images"] = wandb.Image(image_grid_orig)
            wandb.log(log_dict, step=self.step, commit=False)
 

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        local_device = "cuda" if th.cuda.is_available() else "cpu"
        device = local_device if not self.use_distributed else dist_util.dev()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(device)
            micro_cond = {
                k: v[i : i + self.microbatch].to(device)
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                p_guidance=self.p_guidance
            )

            if not self.use_distributed or (last_batch or not self.use_ddp):
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            losses["lr"] = self.opt.param_groups[0]["lr"] 
            losses["lambda"] = self.diffusion.encoder.RD_lambda
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                self.opt.zero_grad()
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self._update_lambda()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self._update_lambda()
        max_grad = max([p.grad.abs().max() for p in self.model.parameters()])
        max_param = max([p.abs().max() for p in self.model.parameters()])
        norm_grad = max([p.grad.norm().max() for p in self.model.parameters()])
        norm_param = max([p.norm().max() for p in self.model.parameters()])
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        wandb.log({
            "max_grad": max_grad,
            "max_param": max_param,
            "norm_grad": norm_grad,
            "norm_param": norm_param
        }, step=self.step, commit=False)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _update_lambda(self):
        if self.step > 500000:
            self.diffusion.encoder.RD_lambda = self.final_RD

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        # frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        # lr = self.lr * (1 - frac_done)
        if self.step > 0 and self.step % self.lr_anneal_steps == 0:
            self.lr = max(self.lr * 0.8, 2e-5)
            for param_group in self.opt.param_groups:
                param_group["lr"] = self.lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if not self.use_distributed or dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                if not rate:
                    filename = f"model{(self.step):06d}_encoder.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}_encoder.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.diffusion.encoder.state_dict(), f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if not self.use_distributed or dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

            if self.use_distributed:
                dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def get_rate(strings, batch_shape, i):
    if len(strings) == 1:
        rate_z = 0
    else:
        rate_z = len(strings[1][i]) * 8 / np.product(batch_shape[2:]) 
    rate_y = len(strings[0][i]) * 8 / np.product(batch_shape[2:]) 
    return rate_z + rate_y 

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        if key not in ["rate", "lr", "lambda"]:
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
