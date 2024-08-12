import model_io
import torch, torch.nn as nn, torch.optim as optim
import model_blocks
import model_training
import data_processing

#input_folder = './datasets/vangogh2photo/train/vangogh'
#output_folder = './datasets/vangogh2photo/train/vangoghAugmented'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

TRAIN_DIR = "./datasets/vangogh2photo/train"
#VAL_DIR = "./datasets/vangogh2photo/val"

#hyperparameter
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0 # loss weight for identity loss
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GENERATOR_B = "./models/genh.pth.tar"
CHECKPOINT_GENERATOR_A = "./models/genz.pth.tar"
CHECKPOINT_DISCRIMINATOR_B = "./models/disch.pth.tar"
CHECKPOINT_DISCRIMINATOR_A = "./models/discz.pth.tar"

loader = data_processing.create_loader(TRAIN_DIR, data_processing.VanGogh2PhotoDataset, BATCH_SIZE, NUM_WORKERS)

disc_B = model_blocks.Discriminator(in_channels=3).to(DEVICE)
disc_A = model_blocks.Discriminator(in_channels=3).to(DEVICE)
gen_A = model_blocks.Generator(img_channels=3, num_residuals=9).to(DEVICE)
gen_B = model_blocks.Generator(img_channels=3, num_residuals=9).to(DEVICE)

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

model_io.load_model(
    CHECKPOINT_GENERATOR_B,
    CHECKPOINT_GENERATOR_A,
    CHECKPOINT_DISCRIMINATOR_B,
    CHECKPOINT_DISCRIMINATOR_A,
    gen_B, gen_A, disc_B, disc_A,
    opt_gen, opt_disc,
    LEARNING_RATE,
    DEVICE,
    LOAD_MODEL
)

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

gen_A_Loss = []
gen_B_Loss = []
cycle_A_Loss = []
cycle_B_Loss = []
G_Loss = []

for epoch in range(NUM_EPOCHS):
   model_training.train_fn(
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
        epoch,
        LAMBDA_CYCLE,
        DEVICE
)

if SAVE_MODEL:
          model_io.save_checkpoint(gen_B, opt_gen, filename=CHECKPOINT_GENERATOR_B)
          model_io.save_checkpoint(gen_A, opt_gen, filename=CHECKPOINT_GENERATOR_A)
          model_io.save_checkpoint(disc_B, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_B)
          model_io.save_checkpoint(disc_A, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_A)
