
from vedo import *

import torch
from nr3d_lib.models.explicits.tetrahedral.dmtet import DMTet, auto_face_normals

if __name__ == "__main__":
    def test_isosurface_1():
        device = torch.device('cuda')
        
        dmtet = DMTet()
        
        pos_nx3 = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]], device=device)
        tet_fx4 = torch.tensor([[0, 1, 2, 3]],  device=device)
        sdf_n = torch.tensor([-1., 1., 1., 1.], device=device)
        verts, faces, uvs, uv_idx = dmtet(pos_nx3, sdf_n, tet_fx4)

        pos_nx3 = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [-1., 0., 0.]], device=device)
        tet_fx4 = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]],  device=device)
        sdf_n = torch.tensor([-1., 1., 1., 1., 1.], device=device)
        verts, faces, uvs, uv_idx = dmtet(pos_nx3, sdf_n, tet_fx4)
        m = Mesh([verts.cpu().numpy(), faces.cpu().numpy()])
        show(m)

    def test_isosurface_2():
        device = torch.device('cuda')
        dmtet = DMTet()
        """
        Test isosurfaces at multiple levels
        """
        pos_nx3 = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]], device=device)
        tet_fx4 = torch.tensor([[0, 1, 2, 3]],  device=device)
        sdf_n = torch.tensor([-1., 1., 2., 3.], device=device)
        
        mesh_actor_list = []
        normals_list = []
        levels = np.linspace(-1, 3, 8)
        
        for level in levels:
            c = color_map(level, 'RdBu', vmin=levels[0], vmax=levels[-1])
            verts, faces, *_ = dmtet(pos_nx3, sdf_n, tet_fx4, level)
            face_normals = auto_face_normals(verts, faces)
            m = Mesh([verts.cpu().numpy(), faces.cpu().numpy()])
            m.lighting('ambient').color(c)
            mesh_actor_list.append(m)
            # Even if the isosurface passes through different edges, there are still only minor floating-point differences between these normals, indicating that it must be a quantity independent of the value of the isosurface.
            normals_list.append(face_normals.tolist())
        print(np.array(normals_list))
        
        # Actor for tetrahedra
        tet_edges_v_idx = tet_fx4[:, dmtet.base_tet_edges]
        tet_edges_v_flat = pos_nx3[tet_edges_v_idx]
        tet_edges_np = tet_edges_v_flat.view(-1, 6, 2, 3).view(-1, 2, 3).data.cpu().numpy()
        tet_actor = Lines(tet_edges_np[:, 0], tet_edges_np[:, 1])
        
        show(*mesh_actor_list, tet_actor)      

    def test_isosurface_3():
        pass

    # test_isosurface_1()
    test_isosurface_2()
    test_isosurface_3()