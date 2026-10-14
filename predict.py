import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as TF

from model import UNet


# =========================================================
# CONFIGURATION
# =========================================================

DATA_ROOT = r"C:\Users\Atul kumar\Documents\project\LoveDA_UNet\dataset"

CHECKPOINT_PATH = r"checkpoints\best_model.pth"

IMAGE_SIZE = 256

NUM_CLASSES = 7


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================================================
# CREATE MODEL
# =========================================================

model = UNet(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("Model loaded successfully.")

print(
    "Validation loss:",
    checkpoint["val_loss"]
)


# =========================================================
# FIND VALIDATION IMAGE
# =========================================================

image_dir = os.path.join(
    DATA_ROOT,
    "Val",
    "Val",
    "Urban",
    "images_png"
)

mask_dir = os.path.join(
    DATA_ROOT,
    "Val",
    "Val",
    "Urban",
    "masks_png"
)


# ---------------------------------------------------------
# Get image files
# ---------------------------------------------------------

image_files = sorted(
    [
        file
        for file in os.listdir(image_dir)
        if file.lower().endswith(".png")
    ]
)


# ---------------------------------------------------------
# Select first image
# ---------------------------------------------------------

filename = image_files[21]

image_path = os.path.join(
    image_dir,
    filename
)

mask_path = os.path.join(
    mask_dir,
    filename
)


print()
print("Image:", filename)


# =========================================================
# LOAD ORIGINAL IMAGE
# =========================================================

original_image = Image.open(
    image_path
).convert("RGB")


# =========================================================
# LOAD GROUND TRUTH MASK
# =========================================================

original_mask = Image.open(
    mask_path
).convert("L")


# =========================================================
# RESIZE IMAGE
# =========================================================

image = TF.resize(
    original_image,
    [
        IMAGE_SIZE,
        IMAGE_SIZE
    ],
    interpolation=TF.InterpolationMode.BILINEAR
)


# =========================================================
# RESIZE MASK
# =========================================================

mask = TF.resize(
    original_mask,
    [
        IMAGE_SIZE,
        IMAGE_SIZE
    ],
    interpolation=TF.InterpolationMode.NEAREST
)


# =========================================================
# IMAGE → TENSOR
# =========================================================

image_tensor = TF.to_tensor(
    image
)


# =========================================================
# NORMALIZE
# =========================================================

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

image_tensor = (
    image_tensor - mean
) / std


# =========================================================
# ADD BATCH DIMENSION
# =========================================================

image_tensor = image_tensor.unsqueeze(
    0
)


# =========================================================
# MOVE TO GPU
# =========================================================

image_tensor = image_tensor.to(
    device
)


# =========================================================
# PREDICTION
# =========================================================

with torch.no_grad():

    output = model(
        image_tensor
    )


# =========================================================
# GET PREDICTED CLASS
# =========================================================

prediction = torch.argmax(
    output,
    dim=1
)


# Remove batch dimension

prediction = prediction.squeeze(
    0
)


# Move to CPU

prediction = prediction.cpu().numpy()


# =========================================================
# CONVERT GROUND TRUTH MASK
# =========================================================

mask = np.array(
    mask,
    dtype=np.int64
)


# LoveDA original labels:
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
# Convert to:
#
# 0 = Background
# 1 = Building
# 2 = Road
# 3 = Water
# 4 = Barren
# 5 = Forest
# 6 = Agriculture
#
# No-data = 255


ground_truth = np.full_like(
    mask,
    255
)

valid = mask > 0

ground_truth[valid] = (
    mask[valid] - 1
)


# =========================================================
# PRINT PREDICTION INFORMATION
# =========================================================

print()
print("Prediction shape:")
print(prediction.shape)

print()
print("Ground truth shape:")
print(ground_truth.shape)

print()
print("Predicted classes:")
print(np.unique(prediction))

print()
print("Ground truth classes:")
print(np.unique(ground_truth))


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture"
]


# =========================================================
# CREATE COLOR MAP
# =========================================================

colors = np.array(
    [
        [0, 0, 0],        # Background
        [255, 0, 0],      # Building
        [255, 255, 0],    # Road
        [0, 0, 255],      # Water
        [139, 69, 19],    # Barren
        [0, 128, 0],      # Forest
        [255, 165, 0]     # Agriculture
    ],
    dtype=np.uint8
)


# =========================================================
# COLORIZE MASK
# =========================================================

def colorize_mask(mask):

    height, width = mask.shape

    colored = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    for class_id in range(NUM_CLASSES):

        colored[
            mask == class_id
        ] = colors[class_id]

    return colored


# =========================================================
# CREATE COLORED MASKS
# =========================================================

colored_prediction = colorize_mask(
    prediction
)

colored_ground_truth = colorize_mask(
    ground_truth
)


# =========================================================
# DISPLAY
# =========================================================

plt.figure(
    figsize=(18, 6)
)


# ---------------------------------------------------------
# Original image
# ---------------------------------------------------------

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    image
)

plt.title(
    "Original Satellite Image"
)

plt.axis(
    "off"
)


# ---------------------------------------------------------
# Ground truth
# ---------------------------------------------------------

plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    colored_ground_truth
)

plt.title(
    "Ground Truth"
)

plt.axis(
    "off"
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    colored_prediction
)

plt.title(
    "U-Net Prediction"
)

plt.axis(
    "off"
)


plt.tight_layout()

plt.show()