import os
import numpy as np
from PIL import Image


# DATASET PATH


DATA_ROOT = r"C:\Users\Atul kumar\Documents\project\LoveDA_UNet\dataset"


# CHECK TRAIN AND VALIDATION

for split in ["Train", "Val"]:

    print("\n" + "=" * 60)
    print(f"{split} DATASET")
    print("=" * 60)

    split_path = os.path.join(
        DATA_ROOT,
        split,
        split
    )

    for scene in ["Urban", "Rural"]:

        print(f"\n--- {scene} ---")

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

        # Check directories

        if not os.path.exists(image_dir):

            print("❌ Image directory not found:")
            print(image_dir)
            continue

        if not os.path.exists(mask_dir):

            print("❌ Mask directory not found:")
            print(mask_dir)
            continue

        # Get files

        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".png")
        ])

        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(".png")
        ])

        # Number of files

        print("Number of images :", len(image_files))
        print("Number of masks  :", len(mask_files))

        # Check filenames

        image_names = set(image_files)
        mask_names = set(mask_files)

        missing_masks = image_names - mask_names
        extra_masks = mask_names - image_names

        if len(missing_masks) == 0:

            print("✅ Every image has a mask")

        else:

            print("❌ Images without masks:")
            print(list(missing_masks)[:10])

        if len(extra_masks) == 0:

            print("✅ No extra masks")

        else:

            print("❌ Extra masks:")
            print(list(extra_masks)[:10])

        # Inspect one image + mask

        if len(image_files) > 0:

            filename = image_files[0]

            image_path = os.path.join(
                image_dir,
                filename
            )

            mask_path = os.path.join(
                mask_dir,
                filename
            )

            image = Image.open(image_path)
            mask = Image.open(mask_path)

            image_array = np.array(image)
            mask_array = np.array(mask)

            print("\nExample file:", filename)

            print(
                "Image shape:",
                image_array.shape
            )

            print(
                "Image dtype:",
                image_array.dtype
            )

            print(
                "Mask shape:",
                mask_array.shape
            )

            print(
                "Mask dtype:",
                mask_array.dtype
            )

            print(
                "Mask unique values:",
                np.unique(mask_array)
            )