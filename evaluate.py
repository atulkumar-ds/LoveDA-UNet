import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import LoveDADataset
from model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = "dataset"

IMAGE_SIZE = 256
BATCH_SIZE = 4

NUM_CLASSES = 7
IGNORE_INDEX = 255

CHECKPOINT_PATH = "checkpoints/best_model.pth"

NUM_WORKERS = 0

CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE")
print("=" * 70)

print("Using:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# DATASET
# ============================================================

val_dataset = LoveDADataset(
    root_dir=ROOT_DIR,
    split="Val",
    image_size=IMAGE_SIZE,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("=" * 70)
print("VALIDATION DATASET")
print("=" * 70)

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Validation batches:",
    len(val_loader)
)

print()


# ============================================================
# MODEL
# ============================================================

model = UNet(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

if not os.path.exists(CHECKPOINT_PATH):

    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


# Handle both possibilities:
# 1. checkpoint is state_dict directly
# 2. checkpoint is a dictionary containing model_state_dict

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        # Assume the dictionary itself is state_dict
        model.load_state_dict(checkpoint)

else:

    model.load_state_dict(checkpoint)


model.eval()

print(
    "Loaded:",
    CHECKPOINT_PATH
)

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# ============================================================
# GLOBAL METRICS
# ============================================================

total_correct = 0
total_valid_pixels = 0


# ============================================================
# VALIDATION
# ============================================================

print("=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

with torch.no_grad():

    for batch_idx, (images, masks) in enumerate(val_loader):

        images = images.to(device)

        masks = masks.to(device)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # ----------------------------------------------------
        # Move to CPU
        # ----------------------------------------------------

        predictions = predictions.cpu().numpy()

        masks = masks.cpu().numpy()

        # ----------------------------------------------------
        # Process each image
        # ----------------------------------------------------

        for pred, target in zip(
            predictions,
            masks
        ):

            # Ignore no-data pixels

            valid = target != IGNORE_INDEX

            pred_valid = pred[valid]

            target_valid = target[valid]

            # ------------------------------------------------
            # Pixel accuracy
            # ------------------------------------------------

            total_correct += np.sum(
                pred_valid == target_valid
            )

            total_valid_pixels += len(
                target_valid
            )

            # ------------------------------------------------
            # Confusion matrix
            #
            # Rows    = Ground Truth
            # Columns = Prediction
            # ------------------------------------------------

            for true_class, predicted_class in zip(
                target_valid,
                pred_valid
            ):

                if (
                    0 <= true_class < NUM_CLASSES
                    and
                    0 <= predicted_class < NUM_CLASSES
                ):

                    confusion_matrix[
                        true_class,
                        predicted_class
                    ] += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 50 == 0
            or
            batch_idx == len(val_loader) - 1
        ):

            print(
                f"Batch "
                f"{batch_idx + 1}/"
                f"{len(val_loader)}"
            )


# ============================================================
# PIXEL ACCURACY
# ============================================================

pixel_accuracy = (
    total_correct /
    total_valid_pixels
)


# ============================================================
# PER-CLASS IoU AND DICE
# ============================================================

class_iou = []

class_dice = []

class_precision = []

class_recall = []


for class_id in range(NUM_CLASSES):

    # --------------------------------------------------------
    # Confusion matrix values
    # --------------------------------------------------------

    true_positive = confusion_matrix[
        class_id,
        class_id
    ]

    false_positive = (
        confusion_matrix[:, class_id].sum()
        - true_positive
    )

    false_negative = (
        confusion_matrix[class_id, :].sum()
        - true_positive
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    if union == 0:

        iou = np.nan

    else:

        iou = (
            true_positive /
            union
        )

    class_iou.append(iou)

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator == 0:

        dice = np.nan

    else:

        dice = (
            2 * true_positive /
            dice_denominator
        )

    class_dice.append(dice)

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:

        precision = np.nan

    else:

        precision = (
            true_positive /
            precision_denominator
        )

    class_precision.append(
        precision
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:

        recall = np.nan

    else:

        recall = (
            true_positive /
            recall_denominator
        )

    class_recall.append(
        recall
    )


# ============================================================
# MEAN METRICS
# ============================================================

mean_iou = np.nanmean(
    class_iou
)

mean_dice = np.nanmean(
    class_dice
)

mean_precision = np.nanmean(
    class_precision
)

mean_recall = np.nanmean(
    class_recall
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print()

print(
    f"{'Class':<15}"
    f"{'IoU':>12}"
    f"{'Dice':>12}"
    f"{'Precision':>14}"
    f"{'Recall':>12}"
)

print("-" * 70)


for i in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[i]:<15}"
        f"{class_iou[i]:>12.4f}"
        f"{class_dice[i]:>12.4f}"
        f"{class_precision[i]:>14.4f}"
        f"{class_recall[i]:>12.4f}"
    )


print("-" * 70)

print(
    f"{'Mean':<15}"
    f"{mean_iou:>12.4f}"
    f"{mean_dice:>12.4f}"
    f"{mean_precision:>14.4f}"
    f"{mean_recall:>12.4f}"
)


# ============================================================
# PIXEL ACCURACY
# ============================================================

print()

print("=" * 70)
print("OVERALL METRICS")
print("=" * 70)

print(
    f"Pixel Accuracy : {pixel_accuracy:.4f}"
)

print(
    f"Mean IoU       : {mean_iou:.4f}"
)

print(
    f"Mean Dice      : {mean_dice:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    "Rows    = Ground Truth"
)

print(
    "Columns = Prediction"
)

print()

# Header

print(
    f"{'':15}",
    end=""
)

for name in CLASS_NAMES:

    print(
        f"{name[:10]:>12}",
        end=""
    )

print()

print("-" * 100)


for i in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[i]:<15}",
        end=""
    )

    for j in range(NUM_CLASSES):

        print(
            f"{confusion_matrix[i, j]:>12,}",
            end=""
        )

    print()


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

normalized_cm = np.zeros_like(
    confusion_matrix,
    dtype=np.float64
)


for i in range(NUM_CLASSES):

    row_sum = confusion_matrix[i].sum()

    if row_sum > 0:

        normalized_cm[i] = (
            confusion_matrix[i] /
            row_sum
        )


# ============================================================
# SAVE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    normalized_cm,
    interpolation="nearest"
)

plt.title(
    "LoveDA Normalized Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Ground Truth Class"
)


# Write values inside cells

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        value = normalized_cm[i, j]

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center"
        )


plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()

print(
    "Confusion matrix saved:"
)

print(
    cm_path
)


# ============================================================
# SAVE METRICS TO TEXT FILE
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_results.txt"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "LoveDA U-Net Evaluation\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Pixel Accuracy : "
        f"{pixel_accuracy:.6f}\n"
    )

    f.write(
        f"Mean IoU       : "
        f"{mean_iou:.6f}\n"
    )

    f.write(
        f"Mean Dice      : "
        f"{mean_dice:.6f}\n\n"
    )

    f.write(
        "Per-Class Metrics\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    for i in range(NUM_CLASSES):

        f.write(
            f"{CLASS_NAMES[i]}\n"
        )

        f.write(
            f"  IoU       : "
            f"{class_iou[i]:.6f}\n"
        )

        f.write(
            f"  Dice      : "
            f"{class_dice[i]:.6f}\n"
        )

        f.write(
            f"  Precision : "
            f"{class_precision[i]:.6f}\n"
        )

        f.write(
            f"  Recall    : "
            f"{class_recall[i]:.6f}\n\n"
        )


print(
    "Metrics saved:"
)

print(
    metrics_path
)

print()

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)