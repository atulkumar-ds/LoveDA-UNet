import numpy as np
import matplotlib.pyplot as plt

from PIL import Image



# CHANGE THESE PATHS


IMAGE_PATH = r"C:\Users\Atul kumar\Documents\project\LoveDA_UNet\dataset\Val\Val\Rural\images_png\2522.png"

MASK_PATH = r"C:\Users\Atul kumar\Documents\project\LoveDA_UNet\dataset\Val\Val\Rural\masks_png\2522.png"



# LOAD IMAGE


image = Image.open(IMAGE_PATH).convert("RGB")

mask = Image.open(MASK_PATH).convert("L")


# Convert to NumPy

image = np.array(image)

mask = np.array(mask)


print("Image shape:", image.shape)
print("Mask shape :", mask.shape)

print("Mask values:", np.unique(mask))



# CREATE COLORED MASK

# RGB colors for LoveDA classes
#
# Original LoveDA labels:
#
# 0 = No-data
# 1 = Background
# 2 = Building
# 3 = Road
# 4 = Water
# 5 = Barren
# 6 = Forest
# 7 = Agriculture


colors = np.array([
    [0, 0, 0],          # 0 - No-data
    [255, 255, 255],    # 1 - Background
    [255, 0, 0],        # 2 - Building
    [255, 255, 0],      # 3 - Road
    [0, 0, 255],        # 4 - Water
    [160, 82, 45],      # 5 - Barren
    [0, 255, 0],        # 6 - Forest
    [255, 165, 0]       # 7 - Agriculture
], dtype=np.uint8)


# Create empty RGB image

colored_mask = np.zeros(
    (mask.shape[0], mask.shape[1], 3),
    dtype=np.uint8
)


# Assign colors

for class_id in range(8):

    colored_mask[mask == class_id] = colors[class_id]


# DISPLAY

plt.figure(figsize=(15, 5))


# Original image

plt.subplot(1, 3, 1)

plt.imshow(image)

plt.title("Original Satellite Image")

plt.axis("off")


# Original mask

plt.subplot(1, 3, 2)

plt.imshow(mask, cmap="gray")

plt.title("Original Mask")

plt.axis("off")


# Colored mask

plt.subplot(1, 3, 3)

plt.imshow(colored_mask)

plt.title("Colored Ground Truth")

plt.axis("off")


plt.tight_layout()

plt.show()