"""
compute and plot all the metrics for the data
"""

import gc
import os
import argparse
import json
import pyiqa
import matplotlib.pyplot as plt
import tikzplotlib
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import multiscale_structural_similarity_index_measure, peak_signal_noise_ratio
from torchvision.io import read_image
from torchvision.utils import save_image
from glob import glob
from tqdm import tqdm
import torch
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser("Compute the metrics")
    parser.add_argument("--dataset", help="dataset that should be compressed to be compared")
    parser.add_argument("--cached", action="store_true", help="use cached results")
    return parser.parse_args()


def add_empty_list(res_dict, codec, metric):
    if codec in res_dict[metric]:
        return res_dict
    res_dict[metric][codec] = []
    return res_dict

def get_patches(images, patch_size=256):
    patches = []
    for image in images:
        for i in range(0, image.shape[1], patch_size):
            for j in range(0, image.shape[2], patch_size):
                patches.append(image[:, i:i+patch_size, j:j+patch_size])
    return torch.stack(patches)

def fid_kodak(fid, compressed, orig):
    images = get_patches(compressed, 256)
    original_images = get_patches(orig, 256)
    os.makedirs("tmp/tmp", exist_ok=True)
    os.makedirs("tmp/tmp/compressed", exist_ok=True)
    os.makedirs("tmp/tmp/orig", exist_ok=True)
    for i, image in enumerate(images):
        save_image(image.float().cpu() / 255, f"tmp/tmp/compressed/{i}.png")
    for i, image in enumerate(original_images):
        save_image(image.float().cpu() / 255, f"tmp/tmp/orig/{i}.png")
    return fid("tmp/tmp/compressed", "tmp/tmp/orig")

def create_metric(metric):
    if metric == "fid":
        metric_f = fid_kodak, pyiqa.create_metric("fid")
    else:
        metric_f = pyiqa.create_metric(metric)
    return metric_f


def compute_value(images, original_images, res_dict, codec, referenced, non_referenced):

    gc.collect()
    torch.cuda.empty_cache()

    for metric in res_dict:
        res_dict = add_empty_list(res_dict, codec, metric)

    images = images.to("cuda")
    for metric, func in non_referenced.items():
        if metric != "fid":
            func.cuda()
            res_dict[metric][codec].append(
                func(images.float()/255).cpu().mean()
            )
        else:
            fid, func = func
            func.cuda()
            res_dict[metric][codec].append(
                fid(func, images, original_images)
            )
        func.cpu()


    for metric, func in referenced.items():
        func.cuda()
        res_dict[metric][codec].append(
            func(
                images.float()/255,
                original_images.float()/255
            ).cpu().mean()
        )
        func.cpu()

    return res_dict

def compute_metrics(qp_range, axs, args, method_name, original_images, cached=False):

    results = {
        "niqe": {},
        "fid": {},
        "musiq": {},
        "brisque": {},
        "dbcnn": {},
        "niqe": {},
        "gmsd": {},
        "ms_ssim": {},
        "nlpd": {},
        "lpips": {},
        "psnr": {},
        "rate": {},
        "ssim": {}
    }

    folder_fp = os.path.join(
        "datasets",
        "references",
        f"{method_name}_{os.path.basename(args.dataset)}",
    )

    folder_diff = os.path.join(
        "tmp",
    )
    folder_hific = os.path.join(
        "datasets",
        "references",
        "hific",
    )

    results_json = os.path.join(folder_fp, "rd.json")
    with open(results_json, "r") as f:
        json_dict = json.load(f)
    results["rate"][method_name] = np.mean([json_dict[key]["rates"] for key in json_dict], axis=0)
    psnrs_prev = np.mean([json_dict[key]["psnrs"] for key in json_dict], axis=0)

    families = [
        # "pretrained_enc_norm",
        "ours_ddim_10",
        "ours_ddim_100",
        "ours_ddpm_1000",
        "ours_ddpm_100"
        # "ours_ddim_1000"
        # "pretrained_enc_norm_ddim_100"
    ]

    colors = {
        "HIFIC": "blue",
        "MSH": "black",
        "ours_ddim_10": "red",
        "ours_ddim_100": "red",
        "yang($\\rho=0$)": "green",
        "yang($\\rho=0.9$)": "green",
        "ours_ddpm_1000": "purple",
        "ours_ddpm_100": "purple"
    }
    
    opacitites = {
        "HIFIC": 1,
        "MSH": 1,
        "ours_ddim_10": 0.6,
        "ours_ddim_100": 1,
        "yang($\\rho=0$)": 0.6,
        "yang($\\rho=0.9$)": 1,
        "ours_ddpm_1000": 1,
        "ours_ddpm_100": 0.6,
    }

    diff = False

    non_referenced_metrics = ["niqe", "fid", "musiq", "brisque", "dbcnn", "niqe"]
    referenced_metrics = ["gmsd", "ssim", "ms_ssim", "nlpd", "lpips", "psnr"]

    referenced = {
        metric: create_metric(metric) for metric in referenced_metrics
    }
    non_referenced = {
        metric: create_metric(metric) for metric in non_referenced_metrics
    }
    if cached:
        with open(os.path.join("cache", "cache.pickle"), "rb") as f:
            results.update(pickle.load(f))


    for family in families:
        if not(cached and family in results["rate"]):
            results["rate"][family] = []
            print(f"Family: {family}")
            family_folder = os.path.join("tmp", family)
            for model_folder in glob(os.path.join(family_folder, "*")):

                with open(os.path.join(model_folder, "rd.json"), "r") as f:
                    rates = json.load(f)
                rate = np.mean(rates["rates"])
                images_diff = read_images(model_folder)
                results = compute_value(
                    images_diff,
                    original_images,
                    results,
                    family,
                    referenced,
                    non_referenced
                )
                results["rate"][family].append(rate)

    if not(cached and method_name in results["rate"]):
        for qp in tqdm(range(1, 6)):
            data_path = os.path.join(folder_fp, str(qp))
            images = read_images(data_path)
            results = compute_value(
                images,
                original_images,
                results,
                method_name,
                referenced,
                non_referenced
            )

    if not(cached and "HIFIC" in results["rate"]):
        for qp in tqdm(range(1, 4)):
            images_hific = read_images(os.path.join(folder_hific, str(qp)))
            results = compute_value(
                images_hific,
                original_images,
                results,
                "HIFIC",
                referenced,
                non_referenced
            )
            paths = glob(os.path.join(folder_hific, str(qp)+"_comp", "*.bin"))
            rate = np.mean([os.stat(path).st_size for path in paths]) * 8
            results["rate"]["HIFIC"].append(
                rate/np.product(images_hific.shape[2:])
            )

    with open(os.path.join("cache", "cache.pickle"), "wb") as f:
        pickle.dump(results, f)
    
    for codec in ["HIFIC", method_name, "yang($\\rho=0$)", "yang($\\rho=0.9$)"] + families:
        rates = results["rate"][codec]
        sort_idx = np.argsort(rates)
        rates = np.array(rates)[sort_idx]
        metrics = [
            "fid",
            "niqe",
            "musiq",
            # "brisque",
            "dbcnn",
            # "niqe",
            "gmsd",
            # "ssim",
            # "ms_ssim",
            "nlpd",
            "lpips",
            "psnr"
        ]
        for met_ind, metric in enumerate(metrics):
            if codec in results[metric] and len(results[metric][codec]) > 0:
                axs[met_ind // 4, met_ind % 4].plot(
                    rates[:len(results[metric][codec])],
                    np.array(results[metric][codec])[
                        sort_idx[:len(results[metric][codec])]
                    ],
                    label=codec.replace("_", " "),
                    color=colors[codec],
                    alpha=opacitites[codec]
                )


def read_images(path):
    arr = []
    filenames = glob(os.path.join(path, "*.png"))
    filenames.sort()
    for filename in filenames:
        arr.append(read_image(filename).unsqueeze(0))
        if arr[-1].shape[2] == 768:
            arr[-1] = arr[-1].permute(0, 1, 3, 2)
    return torch.cat(arr, axis=0)

if __name__ == "__main__":

    args = parse_args()
    cached = args.cached
    original_data_path = os.path.join("datasets", "references", "original")

    original_images = read_images(original_data_path).cuda()
    #increase matplotlib font size
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(2, 4, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 1]})
    axs[0, 0].set_title("FID $\\downarrow$")
    axs[0, 1].set_title("NIQE $\\downarrow$")
    axs[0, 2].set_title("MUSIQ $\\uparrow$")
    # axs[1, 0].set_title("BRISQUE $\\downarrow$")
    axs[0, 3].set_title("DBCNN $\\uparrow$")
    # axs[1, 2].set_title("NIQE $\\downarrow$")
    axs[1, 0].set_title("GMSD $\\downarrow$")
    # axs[2, 1].set_title("SSIM $\\downarrow$")
    # axs[2, 2].set_title("MSSSIM $\\uparrow$")
    axs[1, 1].set_title("NLPD $\\downarrow$")
    axs[1, 2].set_title("LPIPS $\\downarrow$")
    axs[1, 3].set_title("PSNR $\\uparrow$")
    
    compute_metrics(
        [0.0512, 0.0256, 0.0128],
        axs,
        args,
        "MSH",
        original_images,
        cached=cached
    )

    handles, labels = axs[0, 0].get_legend_handles_labels()
    for ax in axs.flatten():
        ax.set_xlabel("Rate (bpp)")
    fig.legend(handles, labels, loc='upper center', ncol=len(labels)//2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("metrics.pdf", bbox_inches='tight')
    tikzplotlib.save("metrics.tex")
    plt.show()
