import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import itertools
import numpy as np
from tqdm import tqdm
import argparse

# Step 1: Define hyperparameters
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 256  # Patch size for training (handles large 1900x5300 images)
    batch_size = 16
    lr = 0.0002
    beta1 = 0.5  # Adam beta1 for GAN stability
    lambda_l1 = 100  # Weight for L1 reconstruction loss
    num_epochs = 200
    data_dir = 'data'  # Root folder with 'broken/' and 'clean/' subdirs
    checkpoint_dir = 'checkpoints'
    sample_dir = 'samples'  # For saving test outputs
    broken_dir = 'data/broken'  # Broken text images
    clean_dir = 'data/clean'    # Clean text images
    restored_dir = 'restored'  # Output folder for restored images

config = Config()               # create config instance

# Create necessary directories
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs(config.sample_dir, exist_ok=True)
os.makedirs(config.data_dir, exist_ok=True)
os.makedirs(config.broken_dir, exist_ok=True)
os.makedirs(config.clean_dir, exist_ok=True)
os.makedirs(config.restored_dir, exist_ok=True)

# Step 2: Custom Dataset for paired broken/clean patches
class DocumentDataset(Dataset):
    def __init__(self, broken_dir, clean_dir, transform=None):
        self.broken_dir = broken_dir
        self.clean_dir = clean_dir
        self.transform = transform
        # Assume images are paired by filename (e.g., img001.png in both)
        self.files = sorted([f for f in os.listdir(broken_dir) if f.endswith(('.png', '.jpg'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        broken_path = os.path.join(self.broken_dir, fname)
        clean_path = os.path.join(self.clean_dir, fname)
        
        # Load full images
        broken_img = Image.open(broken_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Crop random patch to handle large images (focus on text regions)
        w, h = broken_img.size
        crop_w, crop_h = config.img_size, config.img_size
        left = np.random.randint(0, w - crop_w + 1)
        top = np.random.randint(0, h - crop_h + 1)
        broken_patch = broken_img.crop((left, top, left + crop_w, top + crop_h))
        clean_patch = clean_img.crop((left, top, left + crop_w, top + crop_h))
        
        if self.transform:
            broken_patch = self.transform(broken_patch)
            clean_patch = self.transform(clean_patch)
        
        # Concat broken and clean horizontally for input (shape: [C, H, 2W] but we'll split in model)
        input_patch = torch.cat([broken_patch, clean_patch], dim=2)  # [3, 256, 512]
        return {'input': input_patch, 'target': clean_patch}

# Transforms: Normalize to [-1, 1] for GAN stability
transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets (80/20 train/val split; adjust as needed)
all_files = sorted(os.listdir(os.path.join(config.data_dir, 'broken')))
split_idx = int(0.8 * len(all_files))
train_dataset = DocumentDataset(
    os.path.join(config.data_dir, 'broken'),
    os.path.join(config.data_dir, 'clean'),
    transform=transform
)[:split_idx]  # Manual split; use torch.utils.data.Subset for proper indexing
val_dataset = DocumentDataset(
    os.path.join(config.data_dir, 'broken'),
    os.path.join(config.data_dir, 'clean'),
    transform=transform
)[split_idx:]

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

# Step 3: Generator - U-Net architecture for broken -> clean mapping
class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8):
        super(UNetGenerator, self).__init__()
        # Initial conv layer
        self.fc = nn.Sequential(
            nn.Conv2d(input_nc, 64, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling layers
        self.down = nn.ModuleList()
        curr_dim = 64
        for i in range(num_downs):
            self.down.append(nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ))
            curr_dim *= 2
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 3, padding=1),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True)
        )
        curr_dim *= 2
        
        # Upsampling layers with skip connections
        self.up = nn.ModuleList()
        for i in range(num_downs - 1, -1, -1):
            self.up.append(nn.Sequential(
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(curr_dim // 2, curr_dim // 2, 3, padding=1),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ))
            curr_dim //= 2
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, output_nc, 7),
            nn.Tanh()  # Output to [-1, 1]
        )

    def forward(self, x):
        # x: [B, 3, 256, 256] (broken only; clean is target)
        x_broken = x[:, :, :, :config.img_size]  # Split input: first half is broken
        skips = []
        out = self.fc(x_broken)
        for down in self.down:
            out = down(out)
            skips.append(out)
        
        out = self.bottleneck(out)
        skips = skips[::-1]  # Reverse for upsampling
        
        for i, up in enumerate(self.up):
            out = up(out)
            out = torch.cat([out, skips[i]], dim=1)  # Skip connection
        
        return self.final(out)

# Step 4: Discriminator - PatchGAN (70x70 receptive field)
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64):  # Input: broken + fake/real clean concatenated
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: [B, 6, 256, 256]
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)  # Output: [B, 1, 30, 30] patches
        )

    def forward(self, x_broken, x_real_or_fake):
        x = torch.cat([x_broken, x_real_or_fake], dim=1)
        return self.model(x)

# Instantiate models
generator = UNetGenerator().to(config.device)
discriminator = PatchGANDiscriminator().to(config.device)

# Losses
criterion_gan = nn.MSELoss()  # For discriminator (real/fake)
criterion_l1 = nn.L1Loss()    # Reconstruction loss

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

# Step 5: Training Loop
def train_epoch():
    generator.train()
    discriminator.train()
    total_loss_g = 0
    total_loss_d = 0
    
    for batch in tqdm(train_loader, desc='Train'):
        real_broken = batch['input'][:, :, :, :config.img_size].to(config.device)  # [B, 3, 256, 256]
        real_clean = batch['target'].to(config.device)  # [B, 3, 256, 256]
        
        # Train Discriminator
        optimizer_d.zero_grad()
        # Real: broken + real_clean
        pred_real = discriminator(real_broken, real_clean)
        loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
        
        # Fake: broken + generated
        fake_clean = generator(real_broken)
        pred_fake = discriminator(real_broken.detach(), fake_clean.detach())
        loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()
        total_loss_d += loss_d.item()
        
        # Train Generator
        optimizer_g.zero_grad()
        pred_fake = discriminator(real_broken, fake_clean)
        loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
        loss_g_l1 = criterion_l1(fake_clean, real_clean) * config.lambda_l1
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        optimizer_g.step()
        total_loss_g += loss_g.item()
    
    return total_loss_g / len(train_loader), total_loss_d / len(train_loader)

# Main training
for epoch in range(config.num_epochs):
    loss_g, loss_d = train_epoch()
    print(f'Epoch [{epoch+1}/{config.num_epochs}] Loss_G: {loss_g:.4f}, Loss_D: {loss_d:.4f}')
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict()}, 
                   os.path.join(config.checkpoint_dir, f'pix2pix_epoch_{epoch+1}.pth'))
    
    # Sample validation (save one batch)
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i == 0:
                    real_broken = batch['input'][:, :, :, :config.img_size].to(config.device)
                    fake_clean = generator(real_broken)
                    # Denormalize and save (simplified; use torchvision.utils.save_image)
                    utils.save_image(fake_clean * 0.5 + 0.5, os.path.join(config.sample_dir, f'sample_epoch_{epoch+1}.png'), nrow=4)
                    break

print('Training complete! Checkpoints in', config.checkpoint_dir)

# Step 6: Inference Function (for full image restoration)
def restore_image(broken_path, model_path, output_path):
    generator.load_state_dict(torch.load(model_path)['generator'])
    generator.eval()
    broken_img = Image.open(broken_path).convert('RGB')
    w, h = broken_img.size
    transform_inf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Patch-wise restoration (simple overlap; improve with blending)
    restored_patches = []
    step = config.img_size // 2  # Overlap for stitching
    for top in range(0, h - config.img_size + 1, step):
        for left in range(0, w - config.img_size + 1, step):
            patch = broken_img.crop((left, top, left + config.img_size, top + config.img_size))
            patch_t = transform_inf(patch).unsqueeze(0).to(config.device)
            with torch.no_grad():
                restored_patch_t = generator(patch_t)
            restored_patch = transforms.ToPILImage()(restored_patch_t.squeeze(0) * 0.5 + 0.5)
            restored_patches.append((restored_patch, left, top))
    
    # Stitch (basic; use OpenCV for advanced)
    restored = Image.new('RGB', (w, h), (255, 255, 255))
    for patch, l, t in restored_patches:
        restored.paste(patch, (l, t))
    restored.save(output_path)
    print(f'Restored image saved to {output_path}')

# Example usage: restore_image('data/broken/test_img.png', 'checkpoints/pix2pix_epoch_200.pth', 'restored.png')