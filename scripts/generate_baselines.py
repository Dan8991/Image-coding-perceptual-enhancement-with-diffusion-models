from pathlib import Path
from compressai.zoo import bmshj2018_factorized, mbt2018_mean
from improved_diffusion.image_datasets import load_data, _list_image_files_recursively
from torchvision.utils import save_image
import argparse
import os
import numpy as np
import torch as th
import json
from tqdm import tqdm

def get_rate(strings, batch_shape, i):
    if len(strings) == 1:
        rate_z = 0
    else:
        rate_z = len(strings[1][i]) * 8 / np.product(batch_shape[2:]) 
    rate_y = len(strings[0][i]) * 8 / np.product(batch_shape[2:]) 
    return rate_z + rate_y 

def compress(batch, encoder_qp, codec):
    if codec == "FP":
        model = bmshj2018_factorized(quality=encoder_qp, pretrained=True)
    else:
        model = mbt2018_mean(quality=encoder_qp, pretrained=True)

    latent = model.compress(batch)
    rec = model.decompress(latent["strings"], latent["shape"])["x_hat"]
    rate = [get_rate(latent["strings"], batch.shape, 0)]
    mse = th.nn.MSELoss(reduction="none")(rec, batch)
    mse = mse.reshape(len(batch), -1).mean(axis=1)
    psnr = 10 * th.log10(1 / mse).detach().cpu().numpy()
    return rate, psnr, rec

def parse_args():
    parser = argparse.ArgumentParser("Generates images to compute the metrics on")
    parser.add_argument("--dataset", help="dataset that should be compressed to be compared")
    parser.add_argument("--image_size", default=512, help="size of the crop", type=int)
    parser.add_argument("--codec", default="MSH", help="codec to generate baselines for")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    data = load_data(
        data_dir=args.dataset,
        batch_size=1,
        image_size=args.image_size,
        class_cond=True,
        sampling=True,
        deterministic=True,
        augmentation=False
    )


    fp_dict = {}
    codec = args.codec
    filenames = _list_image_files_recursively(args.dataset)
    for sample, filename in tqdm(zip(data, filenames)):

        #fp 
        sample = (sample[0] + 1) / 2
        original_data_path = os.path.join("datasets", "references", "original")
        Path(original_data_path).mkdir(parents=True, exist_ok=True)
        save_image(sample, os.path.join(original_data_path, os.path.basename(filename)))
        fp_data_path = os.path.join("datasets", "references", f"{codec}_{os.path.basename(args.dataset)}")
        Path(fp_data_path).mkdir(parents=True, exist_ok=True)
        fp_dict[filename] = {"psnrs": [], "rates": []}
        for qp in range(1, 9):
            rate, psnr, rec = compress(sample, qp, codec)
            fp_dict[filename]["psnrs"].append(float(psnr[0]))
            fp_dict[filename]["rates"].append(float(rate[0]))
            save_image_path = os.path.join(fp_data_path, str(qp))
            Path(save_image_path).mkdir(parents=True, exist_ok=True)
            save_image(rec, os.path.join(save_image_path, os.path.basename(filename)))

    with open(os.path.join(fp_data_path, "rd.json"),"w") as f:
        json.dump(fp_dict, f, indent=2)



