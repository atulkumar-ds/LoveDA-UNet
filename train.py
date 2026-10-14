import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoveDADataset
from model import UNet


# =========================================================
# CONFIGURATION
# =========================================================

DATA_ROOT = r"C:\Users\Atul kumar\Documents\project\LoveDA_UNet\dataset"

IMAGE_SIZE = 256

BATCH_SIZE = 4

NUM_CLASSES = 7

EPOCHS = 20

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

CHECKPOINT_DIR = "checkpoints"

OUTPUT_DIR = "outputs"


# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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


# =========================================================
# DATASET
# =========================================================

train_dataset = LoveDADataset(
    root_dir=DATA_ROOT,
    split="Train",
    image_size=IMAGE_SIZE,
    augment=True
)

val_dataset = LoveDADataset(
    root_dir=DATA_ROOT,
    split="Val",
    image_size=IMAGE_SIZE,
    augment=False
)


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# =========================================================
# MODEL
# =========================================================

model = UNet(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# =========================================================
# LOSSES
# =========================================================

cross_entropy = nn.CrossEntropyLoss(
    ignore_index=255
)


def dice_loss(
    outputs,
    targets,
    num_classes=7,
    ignore_index=255
):

    # -----------------------------------------------------
    # Convert logits to probabilities
    # -----------------------------------------------------

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    # -----------------------------------------------------
    # Valid pixels
    # -----------------------------------------------------

    valid = (
        targets != ignore_index
    )

    # -----------------------------------------------------
    # Safe target
    # -----------------------------------------------------

    safe_targets = targets.clone()

    safe_targets[
        ~valid
    ] = 0

    # -----------------------------------------------------
    # One-hot target
    # -----------------------------------------------------

    target_one_hot = F.one_hot(
        safe_targets,
        num_classes=num_classes
    )

    target_one_hot = target_one_hot.permute(
        0,
        3,
        1,
        2
    ).float()

    # -----------------------------------------------------
    # Ignore invalid pixels
    # -----------------------------------------------------

    valid = valid.unsqueeze(1)

    probabilities = (
        probabilities * valid
    )

    target_one_hot = (
        target_one_hot * valid
    )

    # -----------------------------------------------------
    # Flatten
    # -----------------------------------------------------

    probabilities = probabilities.reshape(
        probabilities.shape[0],
        probabilities.shape[1],
        -1
    )

    target_one_hot = target_one_hot.reshape(
        target_one_hot.shape[0],
        target_one_hot.shape[1],
        -1
    )

    # -----------------------------------------------------
    # Dice
    # -----------------------------------------------------

    intersection = (
        probabilities * target_one_hot
    ).sum(dim=2)

    denominator = (
        probabilities.sum(dim=2)
        +
        target_one_hot.sum(dim=2)
    )

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    dice = dice.mean()

    return 1.0 - dice


def combined_loss(
    outputs,
    targets
):

    ce = cross_entropy(
        outputs,
        targets
    )

    dice = dice_loss(
        outputs,
        targets,
        NUM_CLASSES
    )

    return (
        0.5 * ce
        +
        0.5 * dice
    )


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# LEARNING RATE SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# =========================================================
# MIXED PRECISION
# =========================================================

use_amp = torch.cuda.is_available()

if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    outputs,
    targets,
    num_classes=7,
    ignore_index=255
):

    predictions = torch.argmax(
        outputs,
        dim=1
    )

    valid = (
        targets != ignore_index
    )

    predictions = predictions[valid]

    targets = targets[valid]

    # -----------------------------------------------------
    # Pixel accuracy
    # -----------------------------------------------------

    if targets.numel() == 0:

        pixel_accuracy = 0.0

    else:

        correct = (
            predictions == targets
        ).sum().item()

        total = targets.numel()

        pixel_accuracy = (
            correct / total
        )

    # -----------------------------------------------------
    # IoU and Dice
    # -----------------------------------------------------

    iou_scores = []

    dice_scores = []

    for class_id in range(
        num_classes
    ):

        pred_class = (
            predictions == class_id
        )

        target_class = (
            targets == class_id
        )

        intersection = (
            pred_class & target_class
        ).sum().item()

        pred_area = (
            pred_class.sum().item()
        )

        target_area = (
            target_class.sum().item()
        )

        union = (
            pred_area
            +
            target_area
            -
            intersection
        )

        # -------------------------------------------------
        # IoU
        # -------------------------------------------------

        if union > 0:

            iou_scores.append(
                intersection / union
            )

        # -------------------------------------------------
        # Dice
        # -------------------------------------------------

        denominator = (
            pred_area
            +
            target_area
        )

        if denominator > 0:

            dice_scores.append(
                (
                    2.0 * intersection
                )
                /
                denominator
            )

    # -----------------------------------------------------
    # Mean IoU
    # -----------------------------------------------------

    if len(iou_scores) > 0:

        mean_iou = (
            sum(iou_scores)
            /
            len(iou_scores)
        )

    else:

        mean_iou = 0.0

    # -----------------------------------------------------
    # Mean Dice
    # -----------------------------------------------------

    if len(dice_scores) > 0:

        mean_dice = (
            sum(dice_scores)
            /
            len(dice_scores)
        )

    else:

        mean_dice = 0.0

    return (
        pixel_accuracy,
        mean_iou,
        mean_dice
    )


# =========================================================
# HISTORY
# =========================================================

train_losses = []

val_losses = []

val_ious = []

val_dices = []

val_accuracies = []


# =========================================================
# BEST MODEL
# =========================================================

best_val_loss = float("inf")


# =========================================================
# TOTAL TRAINING TIMER
# =========================================================

total_training_start = time.time()


# =========================================================
# TRAINING
# =========================================================

for epoch in range(EPOCHS):

    # -----------------------------------------------------
    # Epoch timer
    # -----------------------------------------------------

    epoch_start = time.time()

    print()
    print("=" * 70)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 70)


    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    running_train_loss = 0.0

    train_start = time.time()


    train_bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc="Training",
        unit="batch",
        dynamic_ncols=True
    )


    for batch_idx, (
        images,
        masks
    ) in enumerate(train_bar):

        # -------------------------------------------------
        # Move data
        # -------------------------------------------------

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        # -------------------------------------------------
        # Clear gradients
        # -------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # -------------------------------------------------
        # Forward + loss
        # -------------------------------------------------

        if use_amp:

            with torch.amp.autocast(
                "cuda"
            ):

                outputs = model(
                    images
                )

                loss = combined_loss(
                    outputs,
                    masks
                )

        else:

            outputs = model(
                images
            )

            loss = combined_loss(
                outputs,
                masks
            )

        # -------------------------------------------------
        # Backward
        # -------------------------------------------------

        if use_amp:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        # -------------------------------------------------
        # Running loss
        # -------------------------------------------------

        running_train_loss += (
            loss.item()
        )

        average_train_loss = (
            running_train_loss
            /
            (batch_idx + 1)
        )

        # -------------------------------------------------
        # Progress bar
        # -------------------------------------------------

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{average_train_loss:.4f}"
        )


    train_time = (
        time.time()
        -
        train_start
    )


    # =====================================================
    # TRAIN LOSS
    # =====================================================

    train_loss = (
        running_train_loss
        /
        len(train_loader)
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    running_val_loss = 0.0

    total_accuracy = 0.0

    total_iou = 0.0

    total_dice = 0.0

    validation_batches = 0


    validation_start = time.time()


    val_bar = tqdm(
        val_loader,
        total=len(val_loader),
        desc="Validation",
        unit="batch",
        dynamic_ncols=True
    )


    with torch.no_grad():

        for images, masks in val_bar:

            # -------------------------------------------------
            # Move data
            # -------------------------------------------------

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            if use_amp:

                with torch.amp.autocast(
                    "cuda"
                ):

                    outputs = model(
                        images
                    )

                    loss = combined_loss(
                        outputs,
                        masks
                    )

            else:

                outputs = model(
                    images
                )

                loss = combined_loss(
                    outputs,
                    masks
                )

            # -------------------------------------------------
            # Validation loss
            # -------------------------------------------------

            running_val_loss += (
                loss.item()
            )

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            accuracy, iou, dice = calculate_metrics(
                outputs,
                masks,
                NUM_CLASSES
            )

            total_accuracy += accuracy

            total_iou += iou

            total_dice += dice

            validation_batches += 1

            # -------------------------------------------------
            # Running validation loss
            # -------------------------------------------------

            average_val_loss = (
                running_val_loss
                /
                validation_batches
            )

            val_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{average_val_loss:.4f}"
            )


    validation_time = (
        time.time()
        -
        validation_start
    )


    # =====================================================
    # AVERAGE VALIDATION RESULTS
    # =====================================================

    val_loss = (
        running_val_loss
        /
        validation_batches
    )

    val_accuracy = (
        total_accuracy
        /
        validation_batches
    )

    val_iou = (
        total_iou
        /
        validation_batches
    )

    val_dice = (
        total_dice
        /
        validation_batches
    )


    # =====================================================
    # SCHEDULER
    # =====================================================

    scheduler.step(
        val_loss
    )


    # =====================================================
    # SAVE HISTORY
    # =====================================================

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    val_ious.append(
        val_iou
    )

    val_dices.append(
        val_dice
    )

    val_accuracies.append(
        val_accuracy
    )


    # =====================================================
    # EPOCH TIME
    # =====================================================

    epoch_time = (
        time.time()
        -
        epoch_start
    )


    # =====================================================
    # TOTAL TIME
    # =====================================================

    total_elapsed = (
        time.time()
        -
        total_training_start
    )


    # =====================================================
    # ESTIMATED REMAINING TIME
    # =====================================================

    completed_epochs = epoch + 1

    average_epoch_time = (
        total_elapsed
        /
        completed_epochs
    )

    remaining_epochs = (
        EPOCHS
        -
        completed_epochs
    )

    estimated_remaining = (
        average_epoch_time
        *
        remaining_epochs
    )


    # =====================================================
    # TIME FORMAT FUNCTION
    # =====================================================

    def format_time(seconds):

        seconds = int(seconds)

        hours = seconds // 3600

        minutes = (
            seconds % 3600
        ) // 60

        secs = (
            seconds % 60
        )

        if hours > 0:

            return (
                f"{hours}h "
                f"{minutes}m "
                f"{secs}s"
            )

        elif minutes > 0:

            return (
                f"{minutes}m "
                f"{secs}s"
            )

        else:

            return f"{secs}s"


    # =====================================================
    # PRINT EPOCH RESULTS
    # =====================================================

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print()
    print("=" * 70)
    print(
        f"EPOCH {epoch + 1} RESULTS"
    )
    print("=" * 70)

    print(
        f"Train Loss       : "
        f"{train_loss:.4f}"
    )

    print(
        f"Val Loss         : "
        f"{val_loss:.4f}"
    )

    print(
        f"Pixel Accuracy   : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Mean IoU         : "
        f"{val_iou:.4f}"
    )

    print(
        f"Mean Dice        : "
        f"{val_dice:.4f}"
    )

    print(
        f"Learning Rate    : "
        f"{current_lr:.6f}"
    )

    print()

    print(
        f"Training Time    : "
        f"{format_time(train_time)}"
    )

    print(
        f"Validation Time  : "
        f"{format_time(validation_time)}"
    )

    print(
        f"Epoch Time       : "
        f"{format_time(epoch_time)}"
    )

    print(
        f"Total Elapsed    : "
        f"{format_time(total_elapsed)}"
    )

    print(
        f"Estimated Left   : "
        f"{format_time(estimated_remaining)}"
    )


    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "val_iou":
                    val_iou,

                "val_dice":
                    val_dice,

                "val_accuracy":
                    val_accuracy
            },
            checkpoint_path
        )

        print()
        print(
            "✅ Best model saved:"
        )

        print(
            checkpoint_path
        )


# =========================================================
# SAVE FINAL MODEL
# =========================================================

final_model_path = os.path.join(
    CHECKPOINT_DIR,
    "final_model.pth"
)

torch.save(
    {
        "epoch": EPOCHS,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_losses[-1],

        "val_loss":
            val_losses[-1],

        "val_iou":
            val_ious[-1],

        "val_dice":
            val_dices[-1],

        "val_accuracy":
            val_accuracies[-1]
    },
    final_model_path
)


# =========================================================
# PLOT LOSS
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    label="Train Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    )
)

plt.close()


# =========================================================
# PLOT IOU
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(1, EPOCHS + 1),
    val_ious,
    label="Validation Mean IoU"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "IoU"
)

plt.title(
    "Validation Mean IoU"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "iou_curve.png"
    )
)

plt.close()


# =========================================================
# PLOT DICE
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(1, EPOCHS + 1),
    val_dices,
    label="Validation Dice"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Dice"
)

plt.title(
    "Validation Mean Dice"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "dice_curve.png"
    )
)

plt.close()


# =========================================================
# TOTAL TRAINING TIME
# =========================================================

total_training_time = (
    time.time()
    -
    total_training_start
)


# =========================================================
# TRAINING COMPLETE
# =========================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"Best Validation Loss : "
    f"{best_val_loss:.4f}"
)

print()

print(
    "Best Model:"
)

print(
    os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )
)

print()

print(
    "Final Model:"
)

print(
    final_model_path
)

print()

print(
    f"Total Training Time : "
    f"{format_time(total_training_time)}"
)

print()

print("Saved plots:")

print(
    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "iou_curve.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "dice_curve.png"
    )
)