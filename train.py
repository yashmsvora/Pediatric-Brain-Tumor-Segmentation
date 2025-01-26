import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
import nibabel as nib
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
import gc
import logging
import torchio as tio
from unetr import UNETR
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from torchio import SubjectsLoader
import warnings
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_scheduler
from scipy.spatial.distance import directed_hausdorff

warnings.filterwarnings("ignore")

# Set up logging
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def print_and_log(message):
    print(message)
    logging.info(message)


class BraTSDataset(Dataset):
    def __init__(self, root_dir, fold_idx=None, num_folds=5, train=True, tumor_focus_prob=0.7, min_samples_per_fold=2):
        self.root_dir = Path(root_dir)
        self.train = train
        self.tumor_focus_prob = tumor_focus_prob
        self.weights = []
        
        # Get all patient directories
        self.patient_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        # First, gather patient-level information
        patient_info = []
        for patient_dir in self.patient_dirs:
            preprocessed_files = {
                'flair': list(patient_dir.glob("*-t2f_preprocessed.nii.gz")),
                't1c': list(patient_dir.glob("*-t1c_preprocessed.nii.gz")),
                'seg': list(patient_dir.glob("*-seg_preprocessed.nii.gz"))
            }
            
            if all(len(files) > 0 for files in preprocessed_files.values()):
                seg_data = nib.load(preprocessed_files['seg'][0]).get_fdata().astype(np.int64)
                tumor_voxels = (seg_data > 0).sum()
                total_voxels = seg_data.size
                tumor_ratio = tumor_voxels / total_voxels
                
                patient_info.append({
                    'patient_dir': patient_dir,
                    'tumor_ratio': tumor_ratio,
                    'preprocessed_files': {k: v[0] for k, v in preprocessed_files.items()}
                })
        
        # Compute adaptive binning based on data distribution
        tumor_ratios = np.array([info['tumor_ratio'] for info in patient_info])
        n_bins = min(5, len(patient_info) // min_samples_per_fold)  # Ensure at least min_samples_per_fold per bin
        
        if n_bins < 2:
            # If very few samples, use binary classification (tumor/no-tumor)
            tumor_bins = (tumor_ratios > np.median(tumor_ratios)).astype(int)
        else:
            # Use quantile-based binning to ensure more balanced distribution
            quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
            bin_edges = np.percentile(tumor_ratios, quantiles)
            tumor_bins = np.digitize(tumor_ratios, bin_edges)
        
        # Log stratification information
        bin_counts = np.bincount(tumor_bins)
        print(f"Tumor ratio bin distribution: {bin_counts}")
        
        # Initialize StratifiedKFold with adjusted n_splits if necessary
        actual_folds = min(num_folds, min(bin_counts))
        if actual_folds != num_folds:
            print(f"Reducing number of folds from {num_folds} to {actual_folds} due to limited samples")
        
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        
        # Get patient indices for the specified fold
        all_indices = np.arange(len(patient_info))
        fold_splits = list(skf.split(all_indices, tumor_bins))
        
        if fold_idx is not None:
            train_idx, val_idx = fold_splits[fold_idx % actual_folds]
            current_patients = [patient_info[i] for i in (train_idx if train else val_idx)]
            
            self.dataset_entries = []
            
            for patient in current_patients:
                if train:
                    # Randomly select one variant for training
                    selected_variant = random.choice(['affine', 'elastic', 'preprocessed'])
                
                    # Process only the selected variant
                    flair_files = list(patient['patient_dir'].glob(f"*-t2f_{selected_variant}.nii.gz"))
                    t1c_files = list(patient['patient_dir'].glob(f"*-t1c_{selected_variant}.nii.gz"))
                    seg_files = list(patient['patient_dir'].glob(f"*-seg_{selected_variant}.nii.gz"))
                
                    if flair_files and t1c_files and seg_files:
                        entry = {
                            'patient_dir': patient['patient_dir'],
                            'flair_path': flair_files[0],
                            't1c_path': t1c_files[0],
                            'seg_path': seg_files[0],
                            'variant': selected_variant,
                            'tumor_ratio': patient['tumor_ratio'],
                            'tumor_bin': tumor_bins[all_indices[train_idx if train else val_idx][current_patients.index(patient)]]
                        }
                        self.dataset_entries.append(entry)
                        self.weights.append(patient['tumor_ratio'] + 0.1)
                else:
                    entry = {
                        'patient_dir': patient['patient_dir'],
                        'flair_path': patient['preprocessed_files']['flair'],
                        't1c_path': patient['preprocessed_files']['t1c'],
                        'seg_path': patient['preprocessed_files']['seg'],
                        'variant': 'preprocessed',
                        'tumor_ratio': patient['tumor_ratio'],
                        'tumor_bin': tumor_bins[all_indices[train_idx if train else val_idx][current_patients.index(patient)]]
                    }
                    self.dataset_entries.append(entry)
                    self.weights.append(patient['tumor_ratio'] + 0.1)
            
            # Normalize weights
            if self.weights:
                self.weights = np.array(self.weights) / sum(self.weights)
                
    def __len__(self):
        return len(self.dataset_entries)

    def get_patient_ids(self):
        """Return list of unique patient IDs in this dataset"""
        return list(set(str(entry['patient_dir'].name) for entry in self.dataset_entries))
    
    def get_variant_distribution(self):
        """Return distribution of variants in the dataset"""
        variants = [entry['variant'] for entry in self.dataset_entries]
        return {variant: variants.count(variant) for variant in set(variants)}

    def pad_volume(self, volume):
        # Target shape to pad to
        target_shape = (160, 256, 256)
        current_shape = volume.shape
        
        # Calculate the padding size for each dimension
        pad_size = [
            (
                (target_shape[i] - current_shape[i]) // 2,  # Pad before
                (target_shape[i] - current_shape[i] + 1) // 2  # Pad after
            ) if target_shape[i] > current_shape[i] else (0, 0)
            for i in range(3)
        ]
        
        # Pad the volume with reflect mode
        padded = np.pad(volume, pad_size, mode='reflect')
        
        # Ensure the final shape matches the target shape
        return padded[:target_shape[0], :target_shape[1], :target_shape[2]]

    def safe_crop(self, volume, crop_size, center=None):
        if center is None:
            starts = [
                np.random.randint(0, max(1, s - cs))
                for s, cs in zip(volume.shape, crop_size)
            ]
        else:
            starts = [
                max(0, min(c - cs // 2, s - cs))
                for c, cs, s in zip(center, crop_size, volume.shape)
            ]
        
        starts = [max(0, min(start, vol_size - crop_size[i])) 
                 for i, (start, vol_size) in enumerate(zip(starts, volume.shape))]
        ends = [min(start + size, vol_size) 
               for start, size, vol_size in zip(starts, crop_size, volume.shape)]
        
        slices = tuple(slice(start, end) for start, end in zip(starts, ends))
        volume_crop = volume[slices]
        
        if volume_crop.shape != crop_size:
            pad_size = [(0, max(0, target - current)) 
                       for target, current in zip(crop_size, volume_crop.shape)]
            volume_crop = np.pad(volume_crop, pad_size, mode='reflect')
        
        return volume_crop, slices

    def apply_augmentation(self, volume, seed, seg=False):
        torch.manual_seed(seed)
        
        if(seg):
            transforms_list = [
                # Spatial augmentations (applied to both image and segmentation)
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            ]
        else:
            transforms_list = [
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                tio.RandomNoise(mean=0, std=0.1),
                tio.RandomGamma(log_gamma=(-0.3, 0.3)),
            ]
        
        transform = tio.Compose(transforms_list)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension for TorchIO
        augmented_volume = transform(volume_tensor)
        
        return augmented_volume.squeeze(0).numpy()  # Remove channel dimension

    def __getitem__(self, idx):
        entry = self.dataset_entries[idx]
    
        try:
            # Load and preprocess data
            flair = nib.load(entry['flair_path']).get_fdata().astype(np.float32)
            t1c = nib.load(entry['t1c_path']).get_fdata().astype(np.float32)
            seg = nib.load(entry['seg_path']).get_fdata().astype(np.int64)
    
            # Pad volumes
            flair_padded = self.pad_volume(flair)
            t1c_padded = self.pad_volume(t1c)
            seg_padded = self.pad_volume(seg)
    
            crop_size = (128, 128, 128)
    
            # Tumor-focused cropping
            if np.random.rand() < self.tumor_focus_prob and np.any(seg_padded > 0):
                tumor_coords = np.argwhere(seg_padded > 0)
                if len(tumor_coords) > 0:
                    center = tumor_coords.mean(axis=0).astype(int)
                    flair_crop, crop_slices = self.safe_crop(flair_padded, crop_size, center)
                    t1c_crop = t1c_padded[crop_slices]
                    seg_crop = seg_padded[crop_slices]
                else:
                    flair_crop, crop_slices = self.safe_crop(flair_padded, crop_size)
                    t1c_crop = t1c_padded[crop_slices]
                    seg_crop = seg_padded[crop_slices]
            else:
                flair_crop, crop_slices = self.safe_crop(flair_padded, crop_size)
                t1c_crop = t1c_padded[crop_slices]
                seg_crop = seg_padded[crop_slices]
    
            # Apply augmentations if in training mode
            if self.train:
                seed = np.random.randint(0, 2**32)
                flair_crop = self.apply_augmentation(flair_crop, seed=seed, seg=False)
                t1c_crop = self.apply_augmentation(t1c_crop, seed=seed, seg=False)
                seg_crop = self.apply_augmentation(seg_crop, seed=seed, seg=True)

            # Normalize modalities
            flair_crop = (flair_crop - np.mean(flair_crop)) / (np.std(flair_crop) + 1e-6)
            t1c_crop = (t1c_crop - np.mean(t1c_crop)) / (np.std(t1c_crop) + 1e-6)

            # Convert to tensors with correct dimensions (C, H, W, D)
            # Only add one channel dimension
            flair_tensor = torch.from_numpy(flair_crop).float().unsqueeze(0)  # (1, H, W, D)
            t1c_tensor = torch.from_numpy(t1c_crop).float().unsqueeze(0)      # (1, H, W, D)
            seg_tensor = torch.from_numpy(seg_crop).long()                    # (H, W, D)

            # Stack modalities along channel dimension
            image_tensor = torch.cat([flair_tensor, t1c_tensor], dim=0)       # (2, H, W, D)
            
            # Create TorchIO Subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_tensor),          # Already 4D: (C, H, W, D)
                label=tio.LabelMap(tensor=seg_tensor.unsqueeze(0)),  # Make it 4D: (1, H, W, D)
                patient_id=str(entry['patient_dir'].name)
            )
            
            return subject
    
        except Exception as e:
            print(f"Error processing {entry['patient_dir']} ({entry['variant']}): {str(e)}")
            # Create default subject with correct dimensions
            default_image = torch.zeros((2, 128, 128, 128), dtype=torch.float32)  # 4D: (C, H, W, D)
            default_label = torch.zeros((1, 128, 128, 128), dtype=torch.int64)    # 4D: (1, H, W, D)
            
            return tio.Subject(
                image=tio.ScalarImage(tensor=default_image),
                label=tio.LabelMap(tensor=default_label),
                patient_id=str(entry['patient_dir'].name)
            )


def inference(model, image, device):
    """Run inference using sliding window approach with mixed precision"""
    patch_size = 128
    stride = 96
    
    pad_size = [(8, 8), (8, 8), (2, 3)]
    padded = np.pad(image, pad_size, mode='reflect')
    padded_tensor = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)
    
    # Initialize output volume and count map
    output = torch.zeros((1, model.output_dim, *padded.shape), device=device, dtype=torch.float16)
    count = torch.zeros_like(output, device=device, dtype=torch.float16)
    
    # Process in batches of patches to save memory
    patches = []
    positions = []
    
    for x in range(0, padded.shape[0] - patch_size + 1, stride):
        for y in range(0, padded.shape[1] - patch_size + 1, stride):
            for z in range(0, padded.shape[2] - patch_size + 1, stride):
                patches.append(padded_tensor[..., x:x+patch_size, y:y+patch_size, z:z+patch_size])
                positions.append((x, y, z))
    
    # Process patches in batches
    batch_size = 4
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i+batch_size]
        batch_positions = positions[i:i+batch_size]
        
        # Stack patches into a batch
        batch_input = torch.stack(batch_patches).to(device)
        
        with torch.no_grad():
            batch_pred = model(batch_input)
        
        # Add predictions to output volume
        for pred, (x, y, z) in zip(batch_pred, batch_positions):
            output[..., x:x+patch_size, y:y+patch_size, z:z+patch_size] += pred
            count[..., x:x+patch_size, y:y+patch_size, z:z+patch_size] += 1
        
        # Clear GPU cache
        del batch_input, batch_pred
        torch.cuda.empty_cache()
    
    final_output = output / count
    segmentation = torch.argmax(final_output, dim=1)
    segmentation = segmentation[..., 8:-8, 8:-8, 2:-3]
    
    return segmentation.cpu().numpy()

def calculate_class_weights(dataset: BraTSDataset, num_classes: int, epsilon: float = 1e-6):
    """
    Compute class weights based on voxel counts in segmentation masks.

    Args:
        dataset (BraTSDataset): Dataset containing segmentation masks.
        num_classes (int): Total number of classes in the dataset.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Class weights for each class.
    """
    label_counts = np.zeros(num_classes)

    for entry in dataset.dataset_entries:
        # Load segmentation mask
        seg_path = entry['seg_path']
        seg_data = nib.load(seg_path).get_fdata().astype(np.int64)
        
        # Count voxels for each class
        for c in range(num_classes):
            label_counts[c] += np.sum(seg_data == c)

    # Compute weights: inverse of the class frequency
    total_voxels = label_counts.sum()
    class_weights = total_voxels / (label_counts + epsilon)

    # Optional: Normalize weights to sum to 1
    class_weights /= class_weights.sum()

    return class_weights


class FocalTverskyCELoss(nn.Module):
    def __init__(self, smooth=1.0, alpha=0.3, beta=0.7, gamma=2.0, ce_weight=0.4, tversky_weight=0.6):
        """
        A combined Focal Tversky and Cross-Entropy (CE) loss for segmentation tasks.

        Args:
            smooth (float): Smoothing factor to avoid division by zero in Tversky loss.
            alpha (float): Weight for false positives in Tversky loss.
            beta (float): Weight for false negatives in Tversky loss.
            gamma (float): Focusing parameter for Focal Tversky Loss.
            ce_weight (float): Weight for the CE loss in the total loss.
            tversky_weight (float): Weight for the Tversky loss in the total loss.
        """
        super(FocalTverskyCELoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight

        # Initialize CrossEntropyLoss
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def focal_tversky_loss(self, logits, targets, eps=1e-7):
        """
        Compute Focal Tversky loss.

        Args:
            logits (torch.Tensor): Model output logits.
            targets (torch.Tensor): Ground truth labels.
            eps (float): Small value to prevent division by zero.

        Returns:
            torch.Tensor: Focal Tversky loss value.
        """
        probs = torch.softmax(logits, dim=1).clamp(min=eps, max=1 - eps)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()

        true_positives = torch.sum(probs * targets_one_hot, dim=(2, 3, 4))
        false_positives = torch.sum(probs * (1 - targets_one_hot), dim=(2, 3, 4))
        false_negatives = torch.sum((1 - probs) * targets_one_hot, dim=(2, 3, 4))

        tversky_index = (true_positives + self.smooth) / (
            true_positives + self.alpha * false_positives + self.beta * false_negatives + self.smooth
        )
        focal_tversky_loss = torch.pow(1 - tversky_index, self.gamma)  # Apply focusing parameter
        return focal_tversky_loss.mean()

    def forward(self, logits, targets):
        """
        Compute the total loss as a combination of Focal Tversky and Cross-Entropy loss.

        Args:
            logits (torch.Tensor): Model output logits.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            tuple: Total loss, CE loss, and Focal Tversky loss.
        """
        ce_loss = self.ce_loss(logits.clamp(min=-100, max=100), targets)
        focal_tversky_loss = self.focal_tversky_loss(logits, targets)

        # Check for NaN values and handle appropriately
        if torch.isnan(focal_tversky_loss) or torch.isnan(ce_loss):
            nan_losses = []
            if torch.isnan(ce_loss):
                nan_losses.append("CE Loss")
            if torch.isnan(focal_tversky_loss):
                nan_losses.append("Focal Tversky Loss")
            print(f"Warning: NaN detected in {', '.join(nan_losses)}! Skipping this batch.")
            return None  # Indicate that the current batch should be skipped

        total_loss = self.ce_weight * ce_loss + self.tversky_weight * focal_tversky_loss
        return total_loss, ce_loss, focal_tversky_loss

def multiclass_dice_score(pred, target, num_classes=5, epsilon=1e-6):
    """
    Compute the multi-class Dice Score for segmentation.
    Args:
        pred (torch.Tensor): Predicted class indices, shape (B, H, W, D).
        target (torch.Tensor): Ground truth class indices, shape (B, H, W, D).
        num_classes (int): Number of classes.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: Average Dice Score across all classes.
    """
    # One-hot encode predicted and target tensors
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dice_scores = []

    # Compute Dice Score for each class
    for c in range(num_classes):
        pred_c = pred_one_hot[:, c, ...]  # Predicted mask for class c
        target_c = target_one_hot[:, c, ...]  # Ground truth mask for class c

        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + epsilon) / (pred_c.sum() + target_c.sum() + epsilon)
        dice_scores.append(dice)

    # Return the mean Dice Score across all classes
    return torch.mean(torch.tensor(dice_scores))

def hd95(pred, target):
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')  # Handle empty masks

    # Compute forward and backward distances
    forward_distances = directed_hausdorff(pred_points, target_points)[0]
    backward_distances = directed_hausdorff(target_points, pred_points)[0]

    # Combine distances into a single array
    distances = np.array([forward_distances, backward_distances])

    # Return the 95th percentile
    hd95_value = np.percentile(distances, 95)
    return hd95_value

def save_segmentation_images(data, target, predicted, epoch, batch_idx, base_dir='segmentation_results'):
    """
    Save visualization of brain tumor segmentation results with both T1C and FLAIR modalities.
    """
    # Create directories
    epoch_dir = os.path.join(base_dir, f'epoch_{epoch+1}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Define tumor colors
    tumor_colors = {
        1: (1, 0, 0),    # Enhancing Tumor (ET) - Red
        2: (0, 1, 0),    # Non-Enhancing Tumor (NET) - Green
        3: (1, 1, 0),    # Cystic Component (CC) - Yellow
        4: (0, 1, 1)     # Edema (ED) - Cyan
    }
    
    def create_rgb_mask(segmentation, colors):
        """Create RGB visualization of segmentation mask."""
        rgb = np.zeros((*segmentation.shape, 3), dtype=np.float32)
        for label, color in colors.items():
            mask = segmentation == label
            rgb[mask] = color
        return rgb

    def normalize_intensity(image):
        """Normalize image intensity to [0,1] range."""
        # Handle different shapes
        if len(image.shape) > 2:
            image = image.squeeze()
        
        # Robust normalization
        min_val = np.percentile(image, 1)
        max_val = np.percentile(image, 99)
        
        # Prevent division by zero
        if max_val == min_val:
            return np.zeros_like(image, dtype=np.float32)
        
        normalized = np.clip((image - min_val) / (max_val - min_val), 0, 1)
        
        return normalized
    
    num_samples = min(5, data.shape[0])  # Save up to 5 samples per batch
    
    for idx in range(num_samples):
        # Extract the middle slice along the depth (z-dimension)
        middle_slice = data.shape[2] // 2
        
        # Process T1C and FLAIR slices
        flair_slice = normalize_intensity(data[idx, 1, middle_slice, :, :].detach().cpu().numpy())
        t1c_slice = normalize_intensity(data[idx, 0, middle_slice, :, :].detach().cpu().numpy())
        
        # Process ground truth and predictions
        gt_slice = target[idx, middle_slice, :, :].cpu().numpy()
        pred_slice = predicted[idx, middle_slice, :, :].cpu().numpy()
        
        # Create RGB masks for ground truth and predictions
        gt_rgb = create_rgb_mask(gt_slice, tumor_colors)
        pred_rgb = create_rgb_mask(pred_slice, tumor_colors)
        
        # Plot and save results
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot T1C slice
        axes[0].imshow(t1c_slice, cmap='gray')
        axes[0].set_title("T1C (Axial)")
        axes[0].axis('off')
        
        # Plot FLAIR slice
        axes[1].imshow(flair_slice, cmap='gray')
        axes[1].set_title("FLAIR (Axial)")
        axes[1].axis('off')
        
        # Plot Ground Truth Segmentation
        axes[2].imshow(gt_rgb)
        axes[2].set_title("Ground Truth Segmentation")
        axes[2].axis('off')
        
        # Plot Predicted Segmentation
        axes[3].imshow(pred_rgb)
        axes[3].set_title("Predicted Segmentation")
        axes[3].axis('off')
        
        # Add legend for tumor classes
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=color, label=f'Class {label}')
            for label, color in tumor_colors.items()
        ]
        fig.legend(handles=legend_elements, loc='center right')
        
        # Save the figure with a unique name
        save_path = os.path.join(epoch_dir, f'batch_{batch_idx}_sample_{idx}.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
def train_model(model, train_loader, val_loader, class_weights, fold_idx, num_epochs=250):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if os.path.exists('best_model_1.pth'):
        checkpoint = torch.load('best_model_1.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print_and_log("Loaded the best saved model.")

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)
    
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.02 * num_training_steps)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if isinstance(class_weights, np.ndarray):
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = FocalTverskyCELoss()
    max_grad_norm = 1.0

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10

    train_dice_scores, val_dice_scores = [], []
    train_total_losses, train_ce_losses, train_focal_tversky_losses = [], [], []
    val_total_losses, val_ce_losses, val_focal_tversky_losses = [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_dice = []
        epoch_total_loss, epoch_ce_loss, epoch_focal_tversky_loss = [], [], []

        for batch_idx, batch in enumerate(train_loader):
            data = batch['image'][tio.DATA]
            target = batch['label'][tio.DATA].squeeze(1)
            patient_id = batch['patient_id']

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            try:
                with autocast(device_type="cuda"):
                    output = model(data)
                    loss_result = criterion(output, target)
                    if loss_result is None:
                        continue
                    total_loss, ce_loss, focal_tversky_loss = loss_result

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                predicted = output.argmax(dim=1)

                # Compute metrics
                batch_dice = multiclass_dice_score(predicted, target)

                epoch_dice.append(batch_dice)
                epoch_total_loss.append(total_loss.item())
                epoch_ce_loss.append(ce_loss.item())
                epoch_focal_tversky_loss.append(focal_tversky_loss.item())

                if batch_idx % 10 == 0:
                    print_and_log(f"Epoch {epoch + 1}, Batch {batch_idx}, Patient ID: {patient_id}, Loss: {total_loss.item():.4f}")
                    print_and_log(f"   CE Loss: {ce_loss.item():.4f}, Focal Tversky Loss: {focal_tversky_loss.item():.4f}, Dice Score: {batch_dice:.4f}")
                    
                    # Save segmentation images every few epochs
                    if (epoch + 1) % 5 == 0:
                        save_segmentation_images(
                            data,
                            target,
                            predicted,
                            epoch + 1,
                            batch_idx,
                            base_dir='train_segmentation_results'
                        )
            except RuntimeError as e:
                print_and_log(f"Error in batch {batch_idx}: {str(e)}")
                continue

        train_dice_scores.append(np.mean([d.cpu() for d in epoch_dice]))
        train_total_losses.append(np.mean(epoch_total_loss))
        train_ce_losses.append(np.mean(epoch_ce_loss))
        train_focal_tversky_losses.append(np.mean(epoch_focal_tversky_loss))

        if (epoch + 1) % 5 == 0:
            model.eval()
            val_dice = []
            val_epoch_total_loss, val_epoch_ce_loss, val_epoch_focal_tversky_loss = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    data = batch['image'][tio.DATA]
                    target = batch['label'][tio.DATA].squeeze(1)
                    patient_id = batch['patient_id']
                    
                    data, target = data.to(device), target.to(device)

                    with autocast(device_type="cuda"):
                        output = model(data)
                        val_loss_result = criterion(output, target)
                        if val_loss_result is None:
                            continue

                        val_loss, val_ce_loss, val_focal_tversky_loss = val_loss_result
                        predicted = output.argmax(dim=1)

                        val_dice.append(multiclass_dice_score(predicted, target))
                        val_epoch_total_loss.append(val_loss.item())
                        val_epoch_ce_loss.append(val_ce_loss.item())
                        val_epoch_focal_tversky_loss.append(val_focal_tversky_loss.item())
                        
                        save_segmentation_images(
                            data,
                            target,
                            predicted,
                            epoch + 1,
                            batch_idx,
                            base_dir='validation_segmentation_results'
                        )

            val_dice_scores.append(np.mean([d.cpu() for d in val_dice]))
            val_total_losses.append(np.mean(val_epoch_total_loss))
            val_ce_losses.append(np.mean(val_epoch_ce_loss))
            val_focal_tversky_losses.append(np.mean(val_epoch_focal_tversky_loss))

            print_and_log(f"Validation: Epoch {epoch + 1}, Patient ID: {patient_id}, Dice Score: {val_dice_scores[-1]:.4f}")
            print_and_log(f"   CE Loss: {val_ce_losses[-1]:.4f}, Focal Tversky Loss: {val_focal_tversky_losses[-1]:.4f}, Total Loss: {val_total_losses[-1]:.4f}")

            if val_total_losses[-1] < best_val_loss:
                best_val_loss = val_total_losses[-1]
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print_and_log("Early stopping triggered!")
                    break

    np.save(f'{fold_idx}_train_dice_scores.npy', np.array(train_dice_scores))
    np.save(f'{fold_idx}_val_dice_scores.npy', np.array(val_dice_scores))
    np.save(f'{fold_idx}_train_ce_losses.npy', np.array(train_ce_losses))
    np.save(f'{fold_idx}_val_ce_losses.npy', np.array(val_ce_losses))
    np.save(f'{fold_idx}_train_focal_tversky_losses.npy', np.array(train_focal_tversky_losses))
    np.save(f'{fold_idx}_val_focal_tversky_losses.npy', np.array(val_focal_tversky_losses))
    
    return model
   
def create_data_loaders(root_dir, fold_idx, num_folds=5, batch_size=2, num_workers=4, min_samples_per_fold=2):
    # Create datasets
    train_dataset = BraTSDataset(
        root_dir, 
        fold_idx=fold_idx, 
        num_folds=num_folds, 
        train=True,
        min_samples_per_fold=min_samples_per_fold
    )
    val_dataset = BraTSDataset(
        root_dir, 
        fold_idx=fold_idx, 
        num_folds=num_folds, 
        train=False,
        min_samples_per_fold=min_samples_per_fold
    )

    # Verify patient separation and variant distribution
    train_patients = set(train_dataset.get_patient_ids())
    val_patients = set(val_dataset.get_patient_ids())
    
    # Verify no patient overlap
    assert len(train_patients.intersection(val_patients)) == 0, \
        "Patient overlap detected between train and validation sets!"
    
    # Verify validation only uses preprocessed variant
    val_variants = val_dataset.get_variant_distribution()
    assert list(val_variants.keys()) == ['preprocessed'], \
        f"Validation set contains non-preprocessed variants: {val_variants}"
    
    # Log dataset information
    print_and_log(f"Fold {fold_idx + 1}:")
    print_and_log(f"Number of training patients: {len(train_patients)}")
    print_and_log(f"Number of validation patients: {len(val_patients)}")
    print_and_log(f"Training variants: {train_dataset.get_variant_distribution()}")
    print_and_log(f"Validation variants: {val_dataset.get_variant_distribution()}")
    
    train_loader = tio.SubjectsLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = tio.SubjectsLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    num_classes=5
    class_weights = calculate_class_weights(train_dataset, num_classes)

    return train_loader, val_loader, class_weights

def main():
    # Enable memory efficient options
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_and_log(f"Using device: {device}")
    
    data_root = "train"
    num_folds = 5
    results = []
    
    for fold_idx in range(num_folds):
        print_and_log(f"\nTraining Fold {fold_idx + 1}/{num_folds}")
        
        # Create data loaders for this fold
        train_loader, val_loader, class_weights = create_data_loaders(
            data_root,
            fold_idx=fold_idx,
            num_folds=num_folds,
            batch_size=2,
            num_workers=4
        )
        
        # Initialize model
        model = UNETR(
            img_shape=(128, 128, 128),
            input_dim=2,
            output_dim=5,
            embed_dim=768,
            patch_size=16,
            num_heads=12,
            dropout=0.1
        ).to(device)
        
        # Wrap model for multiple GPUs
        if torch.cuda.device_count() > 1:
            print_and_log(f"Using {torch.cuda.device_count()} GPUs for training.")
            model = nn.DataParallel(model)
            
        # Train model for this fold
        model = train_model(model, train_loader, val_loader, class_weights, fold_idx, num_epochs=250)
        
        output_dir = "models"
        os.makedirs(output_dir, exist_ok=True)

        # Save fold model
        model_path = os.path.join(output_dir, f'model_fold_{fold_idx}.pth')
        torch.save({
            'fold': fold_idx,
            'model_state_dict': model.state_dict(),
        }, model_path)
        
        # Store results for this fold
        results.append({
            'fold': fold_idx,
            'model_path': model_path
        })
    
    print_and_log("Cross-validation completed!")

if __name__ == "__main__":
    main()