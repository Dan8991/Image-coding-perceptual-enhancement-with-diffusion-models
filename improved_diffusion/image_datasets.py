from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import RandomResizedCrop, Compose, RandomHorizontalFlip, CenterCrop
import torch as th
import pickle
import os


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    use_distributed=False,
    augmentation=True,
    sampling=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    if use_distributed:
        import mpi4py as MPI
    else:
        MPI = None
    pickle_path = os.path.join(data_dir, "high_res_images.pickle")
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            all_files = pickle.load(f)
    else:
        all_files = _list_image_files_recursively(data_dir)
        all_files = [
            local_image for local_image in all_files
            if np.min(read_image(local_image).shape[1:]) >= image_size
        ]
        with open(pickle_path, "wb") as f:
            pickle.dump(all_files, f)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=0 if not use_distributed else MPI.COMM_WORLD.Get_rank(),
        num_shards=1 if not use_distributed else MPI.COMM_WORLD.Get_size(),
        sampling=sampling,
        augmentation=augmentation
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        augmentation=True,
        sampling=False
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        if sampling:
            self.augmentation = Compose([])
        elif augmentation:
            self.augmentation = Compose([
                RandomHorizontalFlip(),
                RandomResizedCrop(resolution),
            ])
        else:
            self.augmentation = Compose([
                CenterCrop(resolution)
            ])

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        image = read_image(path)

        if self.augmentation:
            image = self.augmentation(image)

        # grayscale to rgb
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        arr = image.to(th.float32) / 127.5 - 1
        arr = arr.numpy()

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr, out_dict
