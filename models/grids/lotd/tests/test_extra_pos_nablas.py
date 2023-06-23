
import torch

from nr3d_lib.config import ConfigDict
from nr3d_lib.logger import Logger
from nr3d_lib.models.fields.sdf import LoTDSDF, pretrain_sdf_sphere

device = torch.device('cuda:0')

m_noextra = LoTDSDF(
    dtype='half', device=device, 
    encoding_cfg=ConfigDict(
        lotd_cfg=ConfigDict(
            lod_res=[8, 13, 21, 34, 55, 89], 
            lod_n_feats=[2, 2, 2, 2, 2, 2], 
            lod_types=['Dense', 'Dense', 'Dense', 'Dense', 'Hash', 'Hash'],  
            hashmap_size=32768
        ), 
        bounding_size=2.0, 
        param_init_cfg=ConfigDict(
            method='uniform_to_type', 
            bound=1.0e-4
        )
    ), 
    decoder_cfg=ConfigDict(
        D=1, W=64, activation='relu'
    ), 
    geo_init_method='pretrain'
)

m_geo = LoTDSDF(
    dtype='half', device=device, 
    encoding_cfg=ConfigDict(
        lotd_cfg=ConfigDict(
            lod_res=[8, 13, 21, 34, 55, 89], 
            lod_n_feats=[2, 2, 2, 2, 2, 2], 
            lod_types=['Dense', 'Dense', 'Dense', 'Dense', 'Hash', 'Hash'], 
            hashmap_size=32768
        ), 
        bounding_size=2.0, 
        param_init_cfg=ConfigDict(
            method='uniform_to_type', 
            bound=1.0e-4
        )
    ), 
    decoder_cfg=ConfigDict(
        D=1, W=64, activation=ConfigDict(type='softplus', beta=100.0)
    ), 
    extra_pos_embed_cfg=ConfigDict(type='identity'), 
    geo_init_method='pretrain_after_geometric'
)


print(m_noextra)
print(m_geo)

logger = Logger('./dev_test/test_extra_nablas', monitoring='tensorboard', save_imgs=False)
pretrain_sdf_sphere(m_noextra, lr=1.0e-3, num_iters=1000, w_eikonal=0, logger=logger, log_prefix='noextra.')
pretrain_sdf_sphere(m_geo, lr=1.0e-3, num_iters=1000, w_eikonal=0, logger=logger, log_prefix='geo.')


