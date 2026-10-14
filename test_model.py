import torch

from model import UNet


# DEVICE

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# CREATE MODEL

model = UNet(
    num_classes=7
)

model = model.to(device)


# CREATE FAKE INPUT

x = torch.randn(
    4,
    3,
    256,
    256
)

x = x.to(device)


# FORWARD PASS


with torch.no_grad():

    output = model(x)



# PRINT SHAPES

print()
print("Input shape :")
print(x.shape)

print()
print("Output shape:")
print(output.shape)