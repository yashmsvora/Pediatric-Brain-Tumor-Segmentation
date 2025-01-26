import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchio as tio
import glob
from scipy.ndimage import label
from skimage.morphology import remove_small_objects, binary_dilation, binary_erosion, ball
import logging
from datetime import datetime

# Set up logging
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def print_and_log(message):
    print(message)
    logging.info(message)

# Debug Logging for Intermediate Outputs
def log_unique_labels(array, description):
    print(f"{description} - Unique Labels: {np.unique(array)}")
    
class BraTSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.patient_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = os.path.join(self.root_dir, self.patient_dirs[idx])
        
        try:
            flair_path = glob.glob(os.path.join(patient_dir, "*-t2f_preprocessed.nii.gz"))[0]
            t1c_path = glob.glob(os.path.join(patient_dir, "*-t1c_preprocessed.nii.gz"))[0]
            
            flair = nib.load(flair_path).get_fdata().astype(np.float32)
            t1c = nib.load(t1c_path).get_fdata().astype(np.float32)

            # Check if the shapes match
            if flair.shape != t1c.shape:
                raise ValueError(f"Shape mismatch between FLAIR and T1C: {flair.shape} vs {t1c.shape}")

             # Normalize modalities
            flair = (flair - np.mean(flair)) / (np.std(flair) + 1e-6)
            t1c = (t1c - np.mean(t1c)) / (np.std(t1c) + 1e-6)

            # Stack modalities along the channel dimension
            image = np.stack([flair, t1c], axis=0)  # (2, Depth, Height, Width)

            seg_path = glob.glob(os.path.join(patient_dir, "*-seg_preprocessed.nii.gz"))[0]
            seg = nib.load(seg_path).get_fdata().astype(np.int64)

            return {
                "image": torch.tensor(image, dtype=torch.float32),
                "seg": torch.tensor(seg, dtype=torch.long),
                "patient_id": self.patient_dirs[idx]
            }
        except Exception as e:
            print_and_log(f"Error processing patient {self.patient_dirs[idx]}: {e}")
            raise

def load_models(model_paths, device, model_class):
    models = []
    for path in model_paths:
        checkpoint = torch.load(path, map_location=device)
        
        model = model_class(
            img_shape=(128, 128, 128),
            input_dim=2,
            output_dim=5,
            embed_dim=768,
            patch_size=16,
            num_heads=12,
            dropout=0.1).to(device)
        state_dict = checkpoint['model_state_dict']
        
        model_state = model.state_dict()
        for key, value in state_dict.items():
            if key in model_state:
                if model_state[key].shape != value.shape:
                    if len(value.shape) == 5 and len(model_state[key].shape) == 5:
                        min_channels = min(value.shape[1], model_state[key].shape[1])
                        if value.shape[1] > model_state[key].shape[1]:
                            value = value[:, :min_channels]
                        else:
                            model_state[key][:, :min_channels] = value
                    elif len(value.shape) == 1:
                        min_classes = min(len(value), model_state[key].shape[0])
                        if len(value) > model_state[key].shape[0]:
                            value = value[:min_classes]
                        else:
                            model_state[key][:min_classes] = value
                    else:
                        continue
                model_state[key] = value
        
        model.load_state_dict(model_state)
        model.eval()
        models.append(model)
    return models

def save_results(patient_id, original_image, ground_truth, prediction, save_dir):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    tumor_colors = {
        0: (0, 0, 0),    # Background (black)
        1: (1, 0, 0),    # Enhancing Tumor (ET)
        2: (0, 1, 0),    # Non-Enhancing Tumor (NET)
        3: (1, 1, 0),    # Cystic Component (CC)
        4: (0, 1, 1)     # Edema (ED)
    }

    def create_rgb_mask(segmentation, colors):
        segmentation = segmentation.astype(np.int32)
        # Initialize the RGB mask
        rgb = np.zeros((*segmentation.shape, 3), dtype=np.float32)
        
        # Assign colors based on the segmentation labels
        for label, color in colors.items():
            mask = segmentation == label
            rgb[mask] = color
        return rgb

    
    def normalize_intensity(image):
        """Normalize image intensity to [0,1] range."""
        min_val = np.percentile(image, 1)
        max_val = np.percentile(image, 99)
        return np.clip((image - min_val) / (max_val - min_val), 0, 1)
    

    # Find the slice with the maximum tumor coverage
    if ground_truth is not None:
        tumor_sums = ground_truth.sum(axis=(1, 2))  # Sum tumor labels along the width and height
        tumor_slice = tumor_sums.argmax()  # Index of the slice with maximum tumor
    else:
        # If ground_truth is not available, use prediction for tumor detection
        tumor_sums = prediction.sum(axis=(1, 2))
        tumor_slice = tumor_sums.argmax()

    # Extract the identified slice
    flair_slice = normalize_intensity(original_image[0, tumor_slice, :, :])  # FLAIR slice
    t1c_slice = normalize_intensity(original_image[1, tumor_slice, :, :])  # T1C slice
    gt_slice = ground_truth[tumor_slice, :, :] if ground_truth is not None else None
    pred_slice = prediction[tumor_slice, :, :]  # 2D slice of the prediction

    
    # Convert ground truth and prediction to RGB masks
    gt_rgb = create_rgb_mask(gt_slice, tumor_colors)
    pred_rgb = create_rgb_mask(pred_slice, tumor_colors)

    # Plot and save combined results (FLAIR, T1C, Ground Truth, Prediction)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    print_and_log(f"{np.unique(gt_rgb)}, {np.unique(gt_slice)}")

    # FLAIR Original
    axes[0].imshow(flair_slice, cmap='gray')
    axes[0].set_title("Original FLAIR (Axial)")
    axes[0].axis('off')

    # T1C Original
    axes[1].imshow(t1c_slice, cmap='gray')
    axes[1].set_title("Original T1C (Axial)")
    axes[1].axis('off')

    # Ground Truth
    if gt_rgb is not None:
        axes[2].imshow(flair_slice, cmap='gray')  # Using FLAIR as the base
        axes[2].imshow(gt_rgb, alpha=0.5)
        axes[2].set_title("Ground Truth Segmentation")
    else:
        axes[2].imshow(np.zeros_like(flair_slice), cmap='gray')  # Show blank if no ground truth
        axes[2].set_title("Ground Truth Missing")
    axes[2].axis('off')

    # Predicted Segmentation
    axes[3].imshow(flair_slice, cmap='gray')  # Using FLAIR as the base
    axes[3].imshow(pred_rgb, alpha=0.5)
    axes[3].set_title("Predicted Segmentation")
    axes[3].axis('off')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{patient_id}_visualization.png"), bbox_inches='tight', dpi=150)
    plt.close()

def filter_small_regions(segmentation, min_size=100):
    refined_segmentation = np.zeros_like(segmentation)
    for tumor_class in [1, 2, 3, 4]:  # Explicitly list tumor classes
        binary_mask = segmentation == tumor_class
        labeled_mask, num_features = label(binary_mask)
        for component_label in range(1, num_features + 1):
            component_size = (labeled_mask == component_label).sum()
            if component_size >= min_size:
                refined_segmentation[labeled_mask == component_label] = tumor_class
    return refined_segmentation

def apply_morphological_operations(segmentation, iterations=1):
    refined_segmentation = np.zeros_like(segmentation)
    for tumor_class in [1, 2, 3, 4]:
        binary_mask = segmentation == tumor_class
        for _ in range(iterations):
            binary_mask = binary_dilation(binary_mask, ball(1))
            binary_mask = binary_erosion(binary_mask, ball(1))
        refined_segmentation[binary_mask] = tumor_class
    print_and_log(f"Labels after morphological operations: {np.unique(refined_segmentation)}")
    return refined_segmentation

def main(model_class, model_paths, test_data_root, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_and_log(f"Using device: {device}")

    # Load models
    models = load_models(model_paths, device, model_class)

    # Dataset and DataLoader
    test_dataset = BraTSDataset(root_dir=test_data_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    for subject in tqdm(test_loader, desc="Inference on test data"):
        patient_id = subject["patient_id"][0]
        image_tensor = subject["image"].squeeze(0)  # Shape: (C, H, W, D)
        print_and_log(patient_id)
        # Create TorchIO Subject
        subject_dict = tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor.numpy())
        )

        # Sliding window setup with adaptive parameters
        grid_sampler = tio.inference.GridSampler(
            subject_dict, 
            patch_size=(128, 128, 128), 
            patch_overlap=(64, 64, 64)  # Increased overlap for smoother predictions
        )
        patch_loader = DataLoader(grid_sampler, batch_size=1)
        aggregator = tio.inference.GridAggregator(grid_sampler)

        # Perform inference
        for patch in patch_loader:
            locations = patch['location']
            patch_data = patch['image'][tio.DATA]
            
            # Soft voting ensemble prediction
            with torch.no_grad(): 
                softmax_outputs = []
                for model in models:
                    output = model(patch_data.to(device))
                    softmax_outputs.append(torch.softmax(output, dim=1).cpu())

                # Average probabilities across models
                combined_softmax = torch.mean(torch.stack(softmax_outputs), dim=0)
                prediction = combined_softmax.argmax(dim=1)
            aggregator.add_batch(prediction.unsqueeze(0), locations)

        # Aggregate predictions
        final_prediction = aggregator.get_output_tensor().squeeze(0).numpy()

        # Post-processing
        final_prediction = filter_small_regions(final_prediction, min_size=100)
        final_prediction = apply_morphological_operations(final_prediction, iterations=1)

        # Save results
        original_image = image_tensor.numpy()
        ground_truth = subject["seg"].squeeze(0).numpy()
        print_and_log(f"Ground truth: {np.unique(ground_truth)}")                
        print_and_log(f"Predicted: {np.unique(final_prediction)}")                

        save_results(patient_id, original_image, ground_truth, final_prediction, save_dir)

if __name__ == "__main__":
    from unetr import UNETR  # Import your specific model class

    model_paths = [
        "models/model_fold_0.pth", 
        "models/model_fold_1.pth", 
        "models/model_fold_2.pth", 
        "models/model_fold_3.pth", 
        "models/model_fold_4.pth"
    ]
    test_data_root = "test"
    save_dir = "testoutput"
    
    main(UNETR, model_paths, test_data_root, save_dir)