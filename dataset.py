import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class LoveDADataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="Train",
        image_size=256,
        augment=False
    ):
        self.image_size = image_size
        self.augment = augment

        # =================================================
        # DATASET PATH
        # =================================================
        #
        # root_dir/
        # ├── Train/
        # │   └── Train/
        # │       ├── Urban/
        # │       └── Rural/
        # │
        # └── Val/
        #     └── Val/
        #         ├── Urban/
        #         └── Rural/
        #
        # =================================================

        split_path = os.path.join(
            root_dir,
            split,
            split
        )

        # List of (image_path, mask_path)
        self.samples = []

        # =================================================
        # READ URBAN + RURAL
        # =================================================

        for scene in ["Urban", "Rural"]:

            image_dir = os.path.join(
                split_path,
                scene,
                "images_png"
            )

            mask_dir = os.path.join(
                split_path,
                scene,
                "masks_png"
            )

            # ------------------------------------------------
            # Check image folder
            # ------------------------------------------------

            if not os.path.exists(image_dir):

                print(
                    "Image folder not found:",
                    image_dir
                )

                continue

            # ------------------------------------------------
            # Check mask folder
            # ------------------------------------------------

            if not os.path.exists(mask_dir):

                print(
                    "Mask folder not found:",
                    mask_dir
                )

                continue

            # =================================================
            # GET IMAGE FILES
            # =================================================

            image_files = sorted(
                [
                    file
                    for file in os.listdir(image_dir)
                    if file.lower().endswith(".png")
                ]
            )

            # =================================================
            # CREATE IMAGE-MASK PAIRS
            # =================================================

            for filename in image_files:

                image_path = os.path.join(
                    image_dir,
                    filename
                )

                mask_path = os.path.join(
                    mask_dir,
                    filename
                )

                # Only add if corresponding mask exists

                if os.path.exists(mask_path):

                    self.samples.append(
                        (
                            image_path,
                            mask_path
                        )
                    )

        # =================================================
        # PRINT DATASET INFORMATION
        # =================================================

        print(
            f"{split}: "
            f"{len(self.samples)} image-mask pairs"
        )

    # =====================================================
    # NUMBER OF SAMPLES
    # =====================================================

    def __len__(self):

        return len(self.samples)

    # =====================================================
    # GET ONE SAMPLE
    # =====================================================

    def __getitem__(self, index):

        # -------------------------------------------------
        # Get image and mask paths
        # -------------------------------------------------

        image_path, mask_path = self.samples[index]

        # =================================================
        # LOAD IMAGE
        # =================================================

        image = Image.open(
            image_path
        ).convert("RGB")

        # =================================================
        # LOAD MASK
        # =================================================

        mask = Image.open(
            mask_path
        ).convert("L")

        # =================================================
        # RESIZE IMAGE
        # =================================================
        #
        # Bilinear interpolation is good for normal images.
        #
        # =================================================

        image = TF.resize(
            image,
            [
                self.image_size,
                self.image_size
            ],
            interpolation=TF.InterpolationMode.BILINEAR
        )

        # =================================================
        # RESIZE MASK
        # =================================================
        #
        # VERY IMPORTANT:
        #
        # Use NEAREST interpolation for masks.
        #
        # We don't want interpolation to create
        # new class labels.
        #
        # =================================================

        mask = TF.resize(
            mask,
            [
                self.image_size,
                self.image_size
            ],
            interpolation=TF.InterpolationMode.NEAREST
        )

        # =================================================
        # IMAGE → TENSOR
        # =================================================

        image = TF.to_tensor(image)

        # Image shape:
        #
        # [3, 256, 256]

        # =================================================
        # MASK → TENSOR
        # =================================================

        mask = torch.from_numpy(
            np.array(
                mask,
                dtype=np.int64
            )
        )

        # Mask shape:
        #
        # [256, 256]

        # =================================================
        # DATA AUGMENTATION
        # =================================================

        if self.augment:

            # -------------------------------------------------
            # RANDOM HORIZONTAL FLIP
            # -------------------------------------------------

            if torch.rand(1).item() > 0.5:

                image = TF.hflip(image)

                mask = TF.hflip(mask)

            # -------------------------------------------------
            # RANDOM VERTICAL FLIP
            # -------------------------------------------------

            if torch.rand(1).item() > 0.5:

                image = TF.vflip(image)

                mask = TF.vflip(mask)

            # -------------------------------------------------
            # RANDOM 90° ROTATION
            # -------------------------------------------------
            #
            # We use torch.rot90 instead of TF.rotate
            # because our mask is a 2D tensor [H, W].
            #
            # -------------------------------------------------

            if torch.rand(1).item() > 0.5:

                k = int(
                    torch.randint(
                        1,
                        4,
                        (1,)
                    ).item()
                )

                # Image shape:
                # [C, H, W]

                image = torch.rot90(
                    image,
                    k=k,
                    dims=[1, 2]
                )

                # Mask shape:
                # [H, W]

                mask = torch.rot90(
                    mask,
                    k=k,
                    dims=[0, 1]
                )

        # NORMALIZE IMAGE
        
        # ImageNet mean/std.
        

        mean = torch.tensor(
            [
                0.485,
                0.456,
                0.406
            ]
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [
                0.229,
                0.224,
                0.225
            ]
        ).view(
            3,
            1,
            1
        )

        image = (
            image - mean
        ) / std

        # =================================================
        # CONVERT LOVEDA MASK LABELS
        #
        # Original LoveDA:
        #
        # 0 = No-data
        # 1 = Background
        # 2 = Building
        # 3 = Road
        # 4 = Water
        # 5 = Barren
        # 6 = Forest
        # 7 = Agriculture
        #
        # PyTorch:
        #
        # 0 = Background
        # 1 = Building
        # 2 = Road
        # 3 = Water
        # 4 = Barren
        # 5 = Forest
        # 6 = Agriculture
        #
        # Original 0 → 255
        #
        # 255 will be ignored by CrossEntropyLoss.
        #
        # =================================================

        new_mask = torch.full_like(
            mask,
            255
        )

        valid_pixels = mask > 0

        new_mask[valid_pixels] = (
            mask[valid_pixels] - 1
        )

        mask = new_mask

        

        return image, mask