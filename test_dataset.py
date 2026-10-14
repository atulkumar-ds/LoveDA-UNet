from torch.utils.data import DataLoader

from dataset import LoveDADataset



# PATH


DATA_ROOT = r"C:\Users\Atul kumar\Documents\project\LoveDA_UNet\dataset"


# CREATE DATASET

dataset = LoveDADataset(
    root_dir=DATA_ROOT,
    split="Train",
    image_size=256,
    augment=True
)


# PRINT DATASET SIZE

print()
print("Dataset size:", len(dataset))



# GET ONE SAMPLE

image, mask = dataset[0]


print()
print("One sample")
print("----------------------")

print(
    "Image shape:",
    image.shape
)

print(
    "Image dtype:",
    image.dtype
)

print(
    "Mask shape:",
    mask.shape
)

print(
    "Mask dtype:",
    mask.dtype
)

print(
    "Mask values:",
    mask.unique()
)


# DATALOADER

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)


# GET ONE BATCH

images, masks = next(iter(loader))


print()
print("One batch")
print("----------------------")

print(
    "Images shape:",
    images.shape
)

print(
    "Masks shape:",
    masks.shape
)