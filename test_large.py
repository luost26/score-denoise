import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.misc import *
from utils.denoise import *
from models.denoise import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt.pt')
parser.add_argument('--input_xyz', type=str, default='./data/examples/RueMadame/RueMadame_3.txt.xyz')
parser.add_argument('--output_xyz', type=str, default='./data/examples/RueMadame/RueMadame_3_denoised.txt.xyz')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
# Denoiser parameters
parser.add_argument('--cluster_size', type=int, default=30000)
args = parser.parse_args()
seed_all(args.seed)

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = DenoiseNet(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

# Point cloud
pcl = np.loadtxt(args.input_xyz)
pcl = torch.FloatTensor(pcl).to(args.device)

pcl_denoised = denoise_large_pointcloud(
    model=model,
    pcl=pcl,
    cluster_size=args.cluster_size,
    seed=args.seed
)
pcl_denoised = pcl_denoised.cpu().numpy()

print('[INFO] Saving denoised point cloud to: %s' % args.output_xyz)
np.savetxt(args.output_xyz, pcl_denoised, fmt='%.8f')
