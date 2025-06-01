"""

Description: This file houses code to implement dino clustering or duster

2025-05-23

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class Images(Dataset):
    """
    Clean dataset loader for preprocessed IMC data.
    
    Loads our preprocessed, normalized data and applies SCAN's transforms
    (which we've modified to skip normalization).
    """
    
    def __init__(self, root_dir, transform=None, split='train', return_index=False):
        """
        Args:
            root_dir: Path to directory containing train.npz (output of preprocessing.py)
            transform: SCAN's transforms (modified to skip normalization)
            split: 'train' or 'valid'
            return_index: Whether to return image indices
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.return_index = return_index
        
        # Find our preprocessed data
        self.data_path = None
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                npz_path = os.path.join(item_path, 'train.npz')
                if os.path.exists(npz_path):
                    self.data_path = npz_path
                    self.dataset_dir = item_path
                    break
        
        if self.data_path is None:
            raise FileNotFoundError(f"Could not find train.npz in any subdirectory of {root_dir}")
        
        # Load preprocessed data
        data = np.load(self.data_path, allow_pickle=True)
        
        if split == 'train':
            self.images = data['train']
        elif split == 'valid':
            self.images = data['valid']
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Loaded {len(self.images)} preprocessed {split} images")
        
        # Load labels for evaluation
        self.labels_df = self._load_labels()
        

    def _load_labels(self):
        """Load ground truth labels if available"""
        labels_file = f"{self.split}_labels.csv"
        labels_path = os.path.join(self.dataset_dir, labels_file)
        
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            print(f"Loaded {len(labels_df)} ground truth labels")
            return labels_df
        return None
    
    def _estimate_num_classes(self):
        """Estimate number of classes from labels or use default"""
        if self.labels_df is not None and 'scene_id' in self.labels_df.columns:
            return len(self.labels_df['scene_id'].unique())
        return 10
    
    def get_ground_truth_labels(self):
        """Get ground truth labels for evaluation"""
        if self.labels_df is not None and 'scene_id' in self.labels_df.columns:
            return self.labels_df['scene_id'].values
        return None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Load preprocessed image. Two options provided below.
        """
        # Get our preprocessed, normalized image
        image = self.images[idx]  # Already normalized by preprocessing.py
        
        # Convert to tensor, preserving our exact normalization
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()  # CHW format
        else:
            image_tensor = image
        
        final_tensor = image_tensor
        
        
        if self.return_index:
            return final_tensor, 0, idx
        else:
            return final_tensor, 0

class Duster(nn.Module):

    def __init__(self, feature_extractor, centroids):

        super().__init__()

        model = feature_extractor
        centroids = nn.Parameter(centroids) # want to optimize centroids so we include it as a model parameter


    def forward(self, input):
        features = self.model(input)

        return centroids, features

