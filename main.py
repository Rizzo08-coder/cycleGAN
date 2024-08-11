import utils
import os, random, numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import neuralblocks

#input_folder = './datasets/vangogh2photo/train/vangogh'
#output_folder = './datasets/vangogh2photo/train/vangoghAugmented'

#utils.dataset_augmentation(input_folder, output_folder)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

TRAIN_DIR = "./datasets/vangogh2photo/train"
VAL_DIR = "./datasets/vangogh2photo/val"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0 # loss weight for identity loss
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GENERATOR_H = "./models/genh.pth.tar"
CHECKPOINT_GENERATOR_Z = "./models/genz.pth.tar"
CHECKPOINT_DISCRIMINATOR_H = "./models/disch.pth.tar"
CHECKPOINT_DISCRIMINATOR_Z = "./models/discz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


def save_checkpoint(model, optimizer, filename="./models/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fn(
    disc_B, disc_A, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, gen_A_Loss, gen_B_Loss, cycle_A_Loss, cycle_B_Loss, G_Loss, epoch
):
    B_reals = 0
    B_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (a, b) in enumerate(loop):
        a = a.to(DEVICE)
        b = b.to(DEVICE)

        # Train discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_b = gen_B(a)
            D_B_real = disc_B(b)
            D_B_fake = disc_B(fake_b.detach())
            B_reals += D_B_real.mean().item()
            B_fakes += D_B_fake.mean().item()
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            fake_a = gen_A(b)
            D_A_real = disc_A(a)
            D_A_fake = disc_A(fake_a.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            D_loss = (D_B_loss + D_A_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial losses
            D_B_fake = disc_B(fake_b)
            D_A_fake = disc_A(fake_a)
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))

            # cycle losses
            cycle_a = gen_A(fake_b)
            cycle_b = gen_B(fake_a)
            cycle_a_loss = l1(a, cycle_a)
            cycle_b_loss = l1(b, cycle_b)

            # identity losses
            # identity_zebra = gen_Z(zebra)
            # identity_horse = gen_H(horse)
            # identity_zebra_loss = l1(zebra, identity_zebra)
            # identity_horse_loss = l1(horse, identity_horse)

            #gen zebra loss
            gen_A_Loss.append(loss_G_A.item())

            #gen horse loss
            gen_B_Loss.append(loss_G_B.item())

            #cycle zebra loss
            cycle_A_Loss.append(cycle_a_loss.item())

            #cycle horse loss
            cycle_B_Loss.append(cycle_a_loss.item())

            # total loss
            G_loss = (
                loss_G_A
                + loss_G_B
                + cycle_a_loss * LAMBDA_CYCLE
                + cycle_b_loss * LAMBDA_CYCLE
                # + identity_horse_loss * LAMBDA_IDENTITY
                # + identity_zebra_loss * LAMBDA_IDENTITY
            )
            G_Loss.append(G_loss.item())

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx == len(loader) - 1:
          save_image(fake_b * 0.5 + 0.5, f"outputs/b_batch{idx}_epoch{epoch}.png")
          save_image(fake_a * 0.5 + 0.5, f"outputs/a_batch{idx}_epoch{epoch}.png")

        loop.set_postfix(B_real=B_reals / (idx + 1), B_fake=B_fakes / (idx + 1))

gen_A_Loss = []
gen_B_Loss = []
cycle_A_Loss = []
cycle_B_Loss = []
G_Loss = []


disc_B = neuralblocks.Discriminator(in_channels=3).to(DEVICE)
disc_A = neuralblocks.Discriminator(in_channels=3).to(DEVICE)
gen_A = neuralblocks.Generator(img_channels=3, num_residuals=9).to(DEVICE)
gen_B = neuralblocks.Generator(img_channels=3, num_residuals=9).to(DEVICE)

# use Adam Optimizer for both generator and discriminator
opt_disc = optim.Adam(
    list(disc_B.parameters()) + list(disc_A.parameters()),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)

opt_gen = optim.Adam(
     list(gen_A.parameters()) + list(gen_B.parameters()),
     lr=LEARNING_RATE,
     betas=(0.5, 0.999),
)

L1 = nn.L1Loss()
mse = nn.MSELoss()

if LOAD_MODEL:
    load_checkpoint(
        CHECKPOINT_GENERATOR_H,
        gen_B,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_GENERATOR_Z,
        gen_A,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_DISCRIMINATOR_H,
        disc_B,
        opt_disc,
        LEARNING_RATE,
    )
    load_checkpoint(
        CHECKPOINT_DISCRIMINATOR_Z,
        disc_A,
        opt_disc,
        LEARNING_RATE,
    )

dataset = utils.VanGogh2PhotoDataset(
    root_photo=TRAIN_DIR + "/photo",
    root_vangogh=TRAIN_DIR + "/vangogh",
    transform=transforms,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

for epoch in range(NUM_EPOCHS):

    train_fn(
        disc_B,
        disc_A,
        gen_A,
        gen_B,
        loader,
        opt_disc,
        opt_gen,
        L1,
        mse,
        d_scaler,
        g_scaler,
        gen_A_Loss,
        gen_B_Loss,
        cycle_A_Loss,
        cycle_B_Loss,
        G_Loss,
        epoch
)

if SAVE_MODEL:
          save_checkpoint(gen_B, opt_gen, filename=CHECKPOINT_GENERATOR_H)
          save_checkpoint(gen_A, opt_gen, filename=CHECKPOINT_GENERATOR_Z)
          save_checkpoint(disc_B, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_H)
          save_checkpoint(disc_A, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_Z)
