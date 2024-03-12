
import torch
from nr3d_lib.models.accelerations.occgrid import OccGridEma, OccGridEmaBatched

if __name__ == "__main__":
    def unit_test(device=torch.device('cuda')):
        num_pts = 4396
        occ = OccGridEma([32, 32, 32], {'type':'sdf', 's': 64.0}, occ_thre=0.3, device=device)
        
        def dummy_sdf_query(x: torch.Tensor):
            # Expect x to be of range [-1,1]
            return x.norm(dim=-1) - 0.5 # A sphere of radius=0.5
        
        # Init from net
        occ._init_from_net(dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Pure update
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ._step_update_grids(pts, val)
        
        # Gather sample
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ.collect_samples(pts, val)
        # Update after gather sample
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ._step_update_grids(pts, val)
        
        # Update from net (during warmup)
        occ._step_update_from_net(0, dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Update from net
        occ._step_update_from_net(10000, dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Visulization
        from vedo import Volume, show
        occ_val_grid = occ.occ_val_grid.data.cpu().numpy()
        aabb = torch.tensor([[-1,-1,-1], [1,1,1]], dtype=torch.float, device=device)
        spacing = ((aabb[1]-aabb[0]) / occ.resolution).tolist()
        origin = aabb[0].tolist()
        vol = Volume(occ_val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
        vox = vol.legosurface(vmin=occ.occ_thre, vmax=1., boundary=True)
        vox.cmap('GnBu', on='cells', vmin=occ.occ_thre, vmax=1.).add_scalarbar()
        show(vox, __doc__, axes=1, viewup='z').close()
    
    def unit_test_batched(device=torch.device('cuda')):
        num_batch = 7
        num_pts = 4396
        occ = OccGridEmaBatched(num_batch, [32, 32, 32], {'type':'sdf', 's': 64.0}, occ_thre=0.3, device=device)
        
        dummy_radius = torch.empty([num_batch], device=device, dtype=torch.float).uniform_(0.3, 0.7)
        def dummy_sdf_query(x: torch.Tensor, bidx: torch.LongTensor = None):
            # Expect x to be of range [-1,1]
            if bidx is not None:
                radius = dummy_radius[bidx]
                return x.norm(dim=-1) - radius # A sphere of radius
            else:
                return x.norm(dim=-1) - dummy_radius.view(-1, *[1]*(x.dim()-2))
        
        # Init from net
        occ._init_from_net(dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Batched (pure update)
        pts = torch.rand([num_batch, num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ._step_update_grids(pts, None, val)
        
        # Not batched (pure update)
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        bidx = torch.randint(num_batch, [num_pts, ], device=device)
        val = dummy_sdf_query(pts, bidx)
        occ._step_update_grids(pts, bidx, val)
        
        # Batched (gather sample)
        pts = torch.rand([num_batch, num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ.collect_samples(pts, None, val)
        # Batched (update after gather sample)
        pts = torch.rand([num_batch, num_pts, 3], device=device) * 2 - 1
        val = dummy_sdf_query(pts)
        occ._step_update_grids(pts, None, val)

        # Not bathced (gather sample)
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        bidx = torch.randint(num_batch, [num_pts, ], device=device)
        val = dummy_sdf_query(pts, bidx)
        occ.collect_samples(pts, bidx, val)
        # Not batched (update after gather sample)
        pts = torch.rand([num_pts, 3], device=device) * 2 - 1
        bidx = torch.randint(num_batch, [num_pts, ], device=device)
        val = dummy_sdf_query(pts, bidx)
        occ._step_update_grids(pts, bidx, val)
        
        # Update from net (during warmup)
        occ._step_update_from_net(0, dummy_sdf_query, num_steps=4, num_pts=2**18)
        
        # Update from net
        occ._step_update_from_net(10000, dummy_sdf_query, num_steps=4, num_pts=2**18)

        # Batched Visulization
        from vedo import Volume, show
        batched_val_grid = occ.occ_val_grid.data.cpu().numpy()
        vox_actors = []
        aabb = torch.tensor([[-1,-1,-1], [1,1,1]], dtype=torch.float, device=device)
        spacing = ((aabb[1]-aabb[0]) / occ.resolution).tolist()
        for i, occ_val_grid in enumerate(batched_val_grid):
            origin = (aabb[0] + torch.tensor([2. * i, 0., 0.], device=device)).tolist()
            vol = Volume(occ_val_grid, c=['white','b','g','r'], mapper='gpu', origin=origin, spacing=spacing)
            vox = vol.legosurface(vmin=occ.occ_thre, vmax=1., boundary=True)
            vox.cmap('GnBu', on='cells', vmin=occ.occ_thre, vmax=1.).add_scalarbar()
            vox_actors.append(vox)
        show(*vox_actors, __doc__, axes=1, viewup='z').close()
    
    unit_test()
    unit_test_batched()