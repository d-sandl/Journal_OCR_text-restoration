import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision
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
    data_dir = '../data'  # Root folder with 'broken/' and 'clean/' subdirs
    checkpoint_dir = '../checkpoints'
    sample_dir = '../samples'  # For saving test outputs
    broken_dir = '../data/broken'  # Broken text images
    clean_dir = '../data/clean'    # Clean text images
    restored_dir = '../restored'  # Output folder for restored images

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
        
        # Get all broken files
        self.broken_files = sorted([f for f in os.listdir(broken_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Build a map: broken_filename → clean_filename
        self.name_map = {}
        for broken_name in self.broken_files:
            # Extract base name: img0000533_thr80.jpg → img0000533
            base = broken_name.split('_thr')[0]  # works for _thr80, _thr90, etc.
            # Try common extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                clean_candidate = base + ext
                if os.path.exists(os.path.join(clean_dir, clean_candidate)):
                    self.name_map[broken_name] = clean_candidate
                    break
            else:
                print(f"Warning: No clean pair found for {broken_name}")

    def __len__(self):
        return len(self.broken_files)

    def __getitem__(self, idx):
        broken_name = self.broken_files[idx]
        clean_name = self.name_map[broken_name]
        
        broken_path = os.path.join(self.broken_dir, broken_name)
        clean_path = os.path.join(self.clean_dir, clean_name)
        
        broken_img = Image.open(broken_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Random crop same patch from both
        w, h = broken_img.size
        left = np.random.randint(0, w - config.img_size + 1)
        top = np.random.randint(0, h - config.img_size + 1)
        box = (left, top, left + config.img_size, top + config.img_size)
        
        broken_patch = broken_img.crop(box)
        clean_patch = clean_img.crop(box)
        
        if self.transform:
            broken_patch = self.transform(broken_patch)
            clean_patch = self.transform(clean_patch)
        
        return {'broken': broken_patch, 'clean': clean_patch}  # simpler names

# Transforms: Normalize to [-1, 1] for GAN stability
transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# # Load datasets (80/20 train/val split; adjust as needed)
# all_files = sorted(os.listdir(os.path.join(config.data_dir, 'broken')))
# split_idx = int(0.8 * len(all_files))
# train_dataset = DocumentDataset(
#     os.path.join(config.data_dir, 'broken'),
#     os.path.join(config.data_dir, 'clean'),
#     transform=transform
# )[:split_idx]  # Manual split; use torch.utils.data.Subset for proper indexing
# val_dataset = DocumentDataset(
#     os.path.join(config.data_dir, 'broken'),
#     os.path.join(config.data_dir, 'clean'),
#     transform=transform
# )[split_idx:]

# train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

# Load datasets (80/20 train/val split; adjust as needed)
full_dataset = DocumentDataset(
    config.broken_dir,
    config.clean_dir,
    transform=transform
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

# Step 3: Generator - U-Net architecture for broken -> clean mapping
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()

        def down_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def up_block(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers += [nn.Dropout(0.5)]
            return nn.Sequential(*layers)

        # Encoder
        # Input size
        self.down1 = down_block(in_channels, 64)      # 256 → 128
        self.down2 = down_block(64, 128)              # 128 → 64
        self.down3 = down_block(128, 256)             # 64  → 32
        self.down4 = down_block(256, 512)             # 32  → 16
        self.down5 = down_block(512, 512)             # 16  → 8
        self.down6 = down_block(512, 512)             # 8   → 4
        self.down7 = down_block(512, 512)             # 4   → 2
        self.down8 = down_block(512, 512)             # 2   → 1

        # Decoder with skip connections
        self.up1 = up_block(512, 512, dropout=True)
        self.up2 = up_block(1024, 512, dropout=True)
        self.up3 = up_block(1024, 512, dropout=True)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)

        self.final = nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skips
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        out = self.final(torch.cat([u7, d1], dim=1))
        return self.tanh(out)

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
        real_broken = batch['broken'].to(config.device)
        real_clean = batch['clean'].to(config.device)
        
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
                    real_broken = batch['broken'].to(config.device)
                    fake_clean = generator(real_broken)
                    # Denormalize and save (simplified; use torchvision.utils.save_image)
                    # utils.save_image(fake_clean * 0.5 + 0.5, os.path.join(config.sample_dir, f'sample_epoch_{epoch+1}.png'), nrow=4)
                    torchvision.utils.save_image(fake_clean * 0.5 + 0.5, os.path.join(config.sample_dir, f'sample_epoch_{epoch+1}.png'), nrow=4, padding=2, normalize=False)
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