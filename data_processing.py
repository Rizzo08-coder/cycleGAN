import os
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VanGogh2PhotoDataset(Dataset):
    def __init__(self, root_vangogh, root_photo, transform=None):
        self.root_vangogh = root_vangogh
        self.root_photo = root_photo
        self.transform = transform

        self.vangogh_images = os.listdir(root_vangogh)
        self.photo_images = os.listdir(root_photo)
        self.length_dataset = max(len(self.vangogh_images), len(self.photo_images))
        self.vangogh_len = len(self.vangogh_images)
        self.photo_len = len(self.photo_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        vangogh_img = self.vangogh_images[index % self.vangogh_len]
        photo_img = self.photo_images[index % self.photo_len]

        vangogh_path = os.path.join(self.root_vangogh, vangogh_img)
        photo_path = os.path.join(self.root_photo, photo_img)

        vangogh_img = np.array(Image.open(vangogh_path).convert("RGB"))
        photo_img = np.array(Image.open(photo_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=vangogh_img, image0=photo_img)
            vangogh_img = augmentations["image"]
            photo_img = augmentations["image0"]

        return vangogh_img, photo_img

def dataset_augmentation(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if not os.listdir(output_folder):
        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)

            # carico l'immagine originale e la salvo nel dataset aumentato
            image = Image.open(img_path)
            image.save(os.path.join(output_folder, f'{img_name}'))

            # flip orizzontale e salvataggio
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image.save(os.path.join(output_folder, f'flipped_{img_name}'))

            # cambiamento della luminosità e salvataggio
            brightened_image = ImageEnhance.Brightness(image).enhance(1.5)
            brightened_image.save(os.path.join(output_folder, f'brightened_{img_name}'))

            # crop dell'immagine e salvataggio
            cropped_image = image.crop((50, 50, 150, 150))
            cropped_image.save(os.path.join(output_folder, f'cropped_{img_name}'))

    print(f"Grandezza dataset originale: {len(os.listdir(input_folder))}")
    print(f"Grandezza dataset aumentato: {len(os.listdir(output_folder))}")

def __image_processing():
    return A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

def __create_dataset(TRAIN_DIR, dataset_class):
    return dataset_class(
        root_photo=TRAIN_DIR + "/photo",
        root_vangogh=TRAIN_DIR + "/vangogh",
        transform=__image_processing(),
    )

def create_loader(TRAIN_DIR, dataset_class, BATCH_SIZE=1, NUM_WORKERS=2):
    return DataLoader(
        __create_dataset(TRAIN_DIR, dataset_class),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
