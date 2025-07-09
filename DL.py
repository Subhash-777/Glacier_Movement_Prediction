import os
import zipfile
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import glob
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GlacierDatasetProcessor:
    def __init__(self, zip_path, extract_path="./glacier_data"):
        self.zip_path = zip_path
        self.extract_path = extract_path
        self.data_info = {}

    def extract_dataset(self):
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)
            print(f"Dataset extracted to: {self.extract_path}")
            self.inspect_dataset()
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            self.create_sample_data()

    def inspect_dataset(self):
        print("\n=== Dataset Structure ===")
        for root, dirs, files in os.walk(self.extract_path):
            level = root.replace(self.extract_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        self.analyze_file_types()

    def analyze_file_types(self):
        file_types = {}
        image_files = []
        for root, dirs, files in os.walk(self.extract_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
                if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    image_files.append(os.path.join(root, file))
        print(f"\n=== File Types Analysis ===")
        for ext, count in file_types.items():
            print(f"{ext}: {count} files")
        self.data_info['file_types'] = file_types
        self.data_info['image_files'] = image_files
        return file_types, image_files

    def create_sample_data(self):
        print("Creating sample glacier data...")
        os.makedirs(self.extract_path, exist_ok=True)
        num_sequences = 20
        frames_per_sequence = 8
        for seq_id in range(num_sequences):
            seq_dir = os.path.join(self.extract_path, f"glacier_sequence_{seq_id:03d}")
            os.makedirs(seq_dir, exist_ok=True)
            for frame_id in range(frames_per_sequence + 1):
                img = self.generate_synthetic_glacier_frame(frame_id, seq_id)
                img_path = os.path.join(seq_dir, f"frame_{frame_id:03d}.png")
                cv2.imwrite(img_path, img)
        print(f"Sample data created with {num_sequences} sequences and {frames_per_sequence + 1} frames per sequence")
        self.inspect_dataset()

    def generate_synthetic_glacier_frame(self, frame_id, seq_id):
        height, width = 224, 224
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = [240, 248, 255]
        center_x = width // 2 + int(frame_id * 2)
        center_y = height // 2 + int(np.sin(frame_id * 0.5) * 10)
        img = np.ascontiguousarray(img)
        noise = np.random.normal(0, 10, (height, width, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        cv2.ellipse(img, (center_x, center_y), (80, 60), 0, 0, 360, (200, 230, 250), -1)
        return img

class GlacierVideoDataset(Dataset):
    def __init__(self, data_path, sequence_length=8, transform=None, target_transform=None):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_transform = target_transform
        self.sequences = []
        for seq_dir in sorted(glob.glob(os.path.join(data_path, "glacier_sequence_*"))):
            frames = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
            if len(frames) >= sequence_length + 1:
                self.sequences.append(frames)
        if len(self.sequences) == 0:
            raise ValueError("No valid sequences found. Please check the dataset or enable synthetic data generation.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_frames = self.sequences[idx]
        input_frames = []
        for i in range(self.sequence_length):
            frame = cv2.imread(sequence_frames[i])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            input_frames.append(frame)
        target_frame = cv2.imread(sequence_frames[self.sequence_length])
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
        if self.target_transform:
            target_frame = self.target_transform(target_frame)
        input_video = torch.stack(input_frames)
        return input_video, target_frame


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for TimeSformer
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Generate queries, keys, values
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.projection(out)

        return out

class TransformerBlock(nn.Module):
    """
    Transformer block for TimeSformer
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Temporal attention
        x = x + self.attention(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class TimeSformer(nn.Module):
    """
    TimeSformer model for glacier movement prediction
    """

    def __init__(self, img_size=224, patch_size=16, num_frames=8, num_classes=3*224*224,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings
        self.temporal_embed = nn.Parameter(torch.randn(1, num_frames, embed_dim))
        self.spatial_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Decoder for frame prediction
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, num_classes),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.shape

        # Reshape for patch embedding
        x = x.reshape(batch_size * num_frames, channels, height, width)

        # Patch embedding
        x = self.patch_embed(x)  # (batch_size * num_frames, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (batch_size * num_frames, num_patches, embed_dim)

        # Reshape back to include temporal dimension
        x = x.reshape(batch_size, num_frames, self.num_patches, self.embed_dim)

        # Add temporal and spatial embeddings
        x = x + self.spatial_embed.unsqueeze(1)  # Add spatial embedding
        x = x + self.temporal_embed.unsqueeze(2)  # Add temporal embedding

        # Reshape for transformer processing
        x = x.reshape(batch_size, num_frames * self.num_patches, self.embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use class token for prediction
        cls_output = x[:, 0]  # Class token

        # Decode to frame prediction
        prediction = self.decoder(cls_output)
        prediction = prediction.reshape(batch_size, 3, height, width)

        return prediction

class GlacierTrainer:
    """
    Trainer class for TimeSformer model
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self, train_loader, val_loader, epochs):
        """Full training loop"""
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")

            self.scheduler.step()

        print("\nTraining completed!")

    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Curves')

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

class GlacierPredictor:
    """
    Inference class for glacier movement prediction
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, input_sequence):
        """Predict next frame given input sequence"""
        with torch.no_grad():
            input_sequence = input_sequence.to(self.device)
            if input_sequence.dim() == 4:  # Add batch dimension
                input_sequence = input_sequence.unsqueeze(0)

            prediction = self.model(input_sequence)
            return prediction.cpu()

    def visualize_prediction(self, input_sequence, prediction, ground_truth=None):
        """Visualize prediction results"""
        # Convert tensors to numpy arrays
        if torch.is_tensor(input_sequence):
            input_sequence = input_sequence.cpu().numpy()
        if torch.is_tensor(prediction):
            prediction = prediction.cpu().numpy()
        if ground_truth is not None and torch.is_tensor(ground_truth):
            ground_truth = ground_truth.cpu().numpy()

        # Get the last frame from input sequence
        last_frame = input_sequence[0, -1].transpose(1, 2, 0)
        predicted_frame = prediction[0].transpose(1, 2, 0)

        # Create visualization
        fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))

        axes[0].imshow(last_frame)
        axes[0].set_title('Last Input Frame')
        axes[0].axis('off')

        axes[1].imshow(predicted_frame)
        axes[1].set_title('Predicted Next Frame')
        axes[1].axis('off')

        if ground_truth is not None:
            gt_frame = ground_truth.transpose(1, 2, 0)
            axes[2].imshow(gt_frame)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('prediction_visualization.png')
        plt.show()

# main function (simplified snippet to add fallback):
def main():
    config = {
        'zip_path': 'glacier_dataset.zip',
        'sequence_length': 8,
        'batch_size': 4,
        'epochs': 5,
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 6,
        'num_heads': 12
    }

    print("\n1. Processing Dataset...")
    processor = GlacierDatasetProcessor(config['zip_path'])
    processor.extract_dataset()
    processor.create_sample_data()  # Force create if needed

    # Step 2: Data Preparation
    print("\n2. Preparing Data...")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor()
    ])

    # Create dataset
    dataset = GlacierVideoDataset(
        processor.extract_path,
        sequence_length=config['sequence_length'],
        transform=transform,
        target_transform=target_transform
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Step 3: Model Creation
    print("\n3. Creating TimeSformer Model...")

    model = TimeSformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_frames=config['sequence_length'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 4: Training
    print("\n4. Training Model...")

    trainer = GlacierTrainer(model, device)
    trainer.train(train_loader, val_loader, config['epochs'])
    trainer.plot_training_curves()

    # Step 5: Inference
    print("\n5. Running Inference...")

    predictor = GlacierPredictor(model, device)

    # Get a sample from validation set
    sample_data, sample_target = next(iter(val_loader))
    sample_input = sample_data[0:1]  # Take first sample
    sample_gt = sample_target[0:1]

    # Make prediction
    prediction = predictor.predict(sample_input)

    # Visualize results
    predictor.visualize_prediction(sample_input, prediction, sample_gt)

    # Save model
    torch.save(model.state_dict(), 'glacier_timesformer.pth')
    print("\nModel saved as 'glacier_timesformer.pth'")

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()