import torch
from tqdm import tqdm
from torchvision.utils import save_image


def train_fn(
    disc_B, disc_A, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, gen_A_Loss, gen_B_Loss, cycle_A_Loss, cycle_B_Loss, G_Loss, epoch, LAMBDA_CYCLE, DEVICE
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