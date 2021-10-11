import os
import torch
import pytorch3d
import pytorch3d.loss
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import point_cloud_utils as pcu
from tqdm.auto import tqdm
from models.utils import *
from .misc import BlackHole


def load_xyz(xyz_dir):
    all_pcls = {}
    for fn in tqdm(os.listdir(xyz_dir), desc='Loading'):
        if fn[-3:] != 'xyz':
            continue
        name = fn[:-4]
        path = os.path.join(xyz_dir, fn)
        all_pcls[name] = torch.FloatTensor(np.loadtxt(path, dtype=np.float32))
    return all_pcls

def load_off(off_dir):
    all_meshes = {}
    for fn in tqdm(os.listdir(off_dir), desc='Loading'):
        if fn[-3:] != 'off':
            continue
        name = fn[:-4]
        path = os.path.join(off_dir, fn)
        verts, faces = pcu.load_mesh_vf(path)
        verts = torch.FloatTensor(verts)
        faces = torch.LongTensor(faces)
        all_meshes[name] = {'verts': verts, 'faces': faces}
    return all_meshes


class Evaluator(object):

    def __init__(self, output_pcl_dir, dataset_root, dataset, summary_dir, experiment_name, device='cuda', res_gts='8192_poisson', logger=BlackHole()):
        super().__init__()
        self.output_pcl_dir = output_pcl_dir
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.summary_dir = summary_dir
        self.experiment_name = experiment_name
        self.gts_pcl_dir = os.path.join(dataset_root, dataset, 'pointclouds', 'test', res_gts)
        self.gts_mesh_dir = os.path.join(dataset_root, dataset, 'meshes', 'test')
        self.res_gts = res_gts
        self.device = device
        self.logger = logger
        self.load_data()

    def load_data(self):
        self.pcls_up = load_xyz(self.output_pcl_dir)
        self.pcls_high = load_xyz(self.gts_pcl_dir)
        self.meshes = load_off(self.gts_mesh_dir)
        self.pcls_name = list(self.pcls_up.keys())

    def run(self):
        pcls_up, pcls_high, pcls_name = self.pcls_up, self.pcls_high, self.pcls_name
        results = {}
        for name in tqdm(pcls_name, desc='Evaluate'):
            pcl_up = pcls_up[name][:,:3].unsqueeze(0).to(self.device)
            if name not in pcls_high:
                self.logger.warning('Shape `%s` not found, ignored.' % name)
                continue
            pcl_high = pcls_high[name].unsqueeze(0).to(self.device)
            verts = self.meshes[name]['verts'].to(self.device)
            faces = self.meshes[name]['faces'].to(self.device)

            cd = pytorch3d.loss.chamfer_distance(pcl_up, pcl_high)[0].item()
            cd_sph = chamfer_distance_unit_sphere(pcl_up, pcl_high)[0].item()
            hd_sph = hausdorff_distance_unit_sphere(pcl_up, pcl_high)[0].item()

            # p2f = point_to_mesh_distance_single_unit_sphere(
            #     pcl=pcl_up[0],
            #     verts=verts,
            #     faces=faces
            # ).sqrt().mean().item()
            if 'blensor' in self.experiment_name:
                rotmat = torch.FloatTensor(Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()).to(pcl_up[0])
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_up[0].matmul(rotmat.t()),
                    verts=verts,
                    faces=faces
                ).item()
            else:
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_up[0],
                    verts=verts,
                    faces=faces
                ).item()

            results[name] = {
                # 'cd': cd,
                'cd_sph': cd_sph,
                'p2f': p2f,
                # 'hd_sph': hd_sph,
            }

        results = pd.DataFrame(results).transpose()
        res_mean = results.mean(axis=0)
        self.logger.info("\n" + repr(results))
        self.logger.info("\nMean\n" + '\n'.join([
            '%s\t%.12f' % (k, v) for k, v in res_mean.items()
        ]))

        update_summary(
            os.path.join(self.summary_dir, 'Summary_%s.csv' % self.dataset),
            model=self.experiment_name,
            metrics={
                # 'cd(mean)': res_mean['cd'],
                'cd_sph(mean)': res_mean['cd_sph'],
                'p2f(mean)': res_mean['p2f'],
                # 'hd_sph(mean)': res_mean['hd_sph'],
            }
        )


def update_summary(path, model, metrics):
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, sep="\s*,\s*", engine='python')
    else:
        df = pd.DataFrame()
    for metric, value in metrics.items():
        setting = metric
        if setting not in df.columns:
            df[setting] = np.nan
        df.loc[model, setting] = value
    df.to_csv(path, float_format='%.12f')
    return df
