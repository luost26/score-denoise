import random
import torch
from torch.utils.data import Dataset
import pytorch3d.ops
from tqdm.auto import tqdm


def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The second point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_A.size(0)
    seed_idx = torch.randperm(N)[:num_patches]   # (P, )
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)   # (1, P, 3)
    _, _, pat_A = pytorch3d.ops.knn_points(seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True)
    pat_A = pat_A[0]    # (P, M, 3)
    _, _, pat_B = pytorch3d.ops.knn_points(seed_pnts, pcl_B.unsqueeze(0), K=int(ratio*patch_size), return_nn=True)
    pat_B = pat_B[0]
    return pat_A, pat_B
    

class PairedPatchDataset(Dataset):

    def __init__(self, datasets, patch_ratio, on_the_fly=True, patch_size=1000, num_patches=1000, transform=None):
        super().__init__()
        self.datasets = datasets
        self.len_datasets = sum([len(dset) for dset in datasets])
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.on_the_fly = on_the_fly
        self.transform = transform
        self.patches = []
        # Initialize
        if not on_the_fly:
            self.make_patches()

    def make_patches(self):
        for dataset in tqdm(self.datasets, desc='MakePatch'):
            for data in tqdm(dataset):
                pat_noisy, pat_clean = make_patches_for_pcl_pair(
                    data['pcl_noisy'],
                    data['pcl_clean'],
                    patch_size=self.patch_size,
                    num_patches=self.num_patches,
                    ratio=self.patch_ratio
                )   # (P, M, 3), (P, rM, 3)
                for i in range(pat_noisy.size(0)):
                    self.patches.append((pat_noisy[i], pat_clean[i], ))

    def __len__(self):
        if not self.on_the_fly:
            return len(self.patches)
        else:
            return self.len_datasets * self.num_patches


    def __getitem__(self, idx):
        if self.on_the_fly:
            pcl_dset = random.choice(self.datasets)
            pcl_data = pcl_dset[idx % len(pcl_dset)]
            pat_noisy, pat_clean = make_patches_for_pcl_pair(
                pcl_data['pcl_noisy'],
                pcl_data['pcl_clean'],
                patch_size=self.patch_size,
                num_patches=1,
                ratio=self.patch_ratio
            )
            data = {
                'pcl_noisy': pat_noisy[0],
                'pcl_clean': pat_clean[0]
            }
        else:
            data = {
                'pcl_noisy': self.patches[idx][0].clone(), 
                'pcl_clean': self.patches[idx][1].clone(),
            }
        if self.transform is not None:
            data = self.transform(data)
        return data
