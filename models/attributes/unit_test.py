import torch
from nr3d_lib.models.attributes import *

if __name__ == "__main__":
    from icecream import ic
    def stat(expr):
        ic(type(expr), expr)

    def test():
        a = RotationMat3x3.random()
        b = RotationQuaternion(learnable=True)
        c = RotationAxisAngle(learnable=True)
        c.tensor.data[2] = -0.5
        d = TransformMat4x4(TransformRT(rot=a, trans=Translation()).mat_4x4(), learnable=True, dtype=torch.float)
        stat(a.inv())
        stat(b.inv())
        stat(RotationMat3x3(c.mat_3x3()).inv())
        stat(a(x=torch.randn([100,7,3])))
        stat(d.inv())
        stat(d.rotate(torch.randn(7,3), inv=True))
        stat(d(torch.randn(7,3), inv=True))

    def test_nested():
        rot = RotationMat3x3(torch.randn([7,3,3]))
        trans = Translation(torch.randn([7,3]))
        pose = TransformRT(rot=rot, trans=trans)
        stat(pose.mat_3x4())
        stat(pose[:3].mat_4x4())
        ex1 = TransformExpSE3(w=Vector_3(torch.randn([3]), learnable=True), v=Vector_3(torch.randn([3]), learnable=True), theta=Scalar(torch.randn([]), learnable=True))
        stat(ex1.translation())
        ex2 = TransformExpSE3(w=Vector_3(torch.randn([7,3]), learnable=True), v=Vector_3(torch.randn([7,3]), learnable=True), theta=Scalar(torch.randn([7]), learnable=True))
        stat(ex2.rotation())
        stat(ex2.translation())
        stat(ex2.rotation_translation())
        stat(ex2.rotate(torch.randn(7,3)))
        stat(ex2.rotate(torch.randn(7,3), inv=True))
        stat(ex2(torch.randn(7,3)))

    def test_focal():
        # fxy_r = FocalRatioExp(torch.zeros([7,2]), learnable=True)
        fxy_r = FocalRatioExp.from_fov(size=[7,], learnable=True)
        intrs = PinholeCameraHWFxyRatio(fxy_r=fxy_r, H=100, W=100)
        from torch.optim import Adam
        optimizer = Adam(intrs.parameters(), lr=0.1)
        stat(intrs.mat_4x4())
        optimizer.zero_grad()
        intrs[2:5].mat_4x4().mean().backward()
        optimizer.step()
        # NOTE: focal is updated by gradient descent
        stat(intrs.mat_4x4())
        
        x = torch.randn(7,3)
        ic(intrs)
        ic(intrs.proj(x))
        a = PinholeCameraMatHW(mat=intrs.mat_4x4().data, H=100, W=100)
        ic(a)
        ic(a.proj(x))
        fx, fy = a.focal().movedim(-1,0)
        b = PinholeCameraHWFxy(fx=fx, fy=fy, H=100, W=100)
        ic(b)
        ic(b.proj(x))
        c = PinholeCameraHWF(f=fx, H=100, W=100)
        ic(c)
        ic(c.proj(x))
        
    def test_stack():
        quats = [RotationQuaternion.random() for _ in range(7)]
        quats_stack = RotationQuaternion.stack(quats)
        ic(quats_stack.tensor)
        
        rts = [TransformRT(rot=RotationQuaternion.random(), trans=Translation(torch.randn([3,]))) for _ in range(7)]
        rts_stack = TransformRT.stack(rts)
        ic(rts_stack.mat_3x4())

    def test_refine():
        a = RotationRefinedMul(attr0=RotationMat3x3.random(), delta=RotationQuaternion(learnable=True), dtype=torch.float)
        ic(a.mat_3x3())

    def test_repr():
        s = Scalar(torch.randn(7,))
        stat(repr(s))

    def test_batched(device=torch.device('cuda')):
        from icecream import ic
        prefix = [7,13]
        # prefix = ()
        pts = torch.randn([1,1,  100,70,  3], device=device, dtype=torch.float)
        # pts = torch.randn([100,3], device=device, dtype=torch.float)
        
        intr = PinholeCameraHWF(
            f=Scalar(torch.full(prefix, 200.0, device=device, dtype=torch.float)), 
            H=Scalar(torch.full(prefix, 200.0, device=device, dtype=torch.float)), 
            W=Scalar(torch.full(prefix, 200.0, device=device, dtype=torch.float)))
        ic(intr)
        
        # NOTE: First index on the 0th dim, then index on the 1st dim
        intr1 = intr[2]
        intr11 = intr1[8]
        ic(intr1)
        ic(intr11)
        u, v, d = intr.proj(pts)
        ic(u.shape)
        
        m1 = RotationMat3x3(torch.randn([*prefix,3,3], device=device, dtype=torch.float))
        ic(m1)
        ic(m1(pts).shape)
        
        m2 = RotationQuaternion(torch.randn([*prefix, 4], device=device, dtype=torch.float))
        ic(m2)
        ic(m2(pts).shape)
        
        m3 = RotationAxisAngle(torch.randn([*prefix, 3], device=device, dtype=torch.float))
        ic(m3)
        ic(m3(pts).shape)
        
        m4 = Rotation6D(torch.randn([*prefix, 6], device=device, dtype=torch.float))
        ic(m4)
        ic(m4(pts).shape)
        
        m5 = TransformRT(rot=m2, trans=Translation(torch.randn([*prefix,3], device=device, dtype=torch.float)))
        ic(m5)
        ic(m5(pts).shape)
        
        m6 = TransformMat4x4(torch.randn([*prefix, 4,4], device=device, dtype=torch.float))
        ic(m6(pts).shape)
        
        m7 = TransformExpSE3(
            w=torch.randn([*prefix,3], device=device, dtype=torch.float), 
            v=torch.randn([*prefix,3], device=device, dtype=torch.float), 
            theta=torch.randn([*prefix], device=device, dtype=torch.float))
        ic(m7)
        ic(m7(pts).shape)
        
        m8 = PinholeCameraMatHW(
            mat=torch.randn([*prefix,4,4], device=device, dtype=torch.float), 
            H=200., W=200., 
            device=device, dtype=torch.float
        )
        ic(m8)
        ic(m8.lift(u, v, d).shape)
        
        m9_1 = PinholeCameraHWFxy(
            fx=torch.randn([*prefix], device=device, dtype=torch.float),
            fy=torch.randn([*prefix], device=device, dtype=torch.float),
            H=torch.randn([*prefix], device=device, dtype=torch.float),
            W=torch.randn([*prefix], device=device, dtype=torch.float),
        )
        ic(m9_1)
        ic(m9_1.lift(u, v, d).shape)
        
        m9_2 = PinholeCameraHWFxy(
            fx=torch.randn([*prefix], device=device, dtype=torch.float),
            fy=torch.randn([*prefix], device=device, dtype=torch.float),
            H=100.,
            W=100., 
            device=device, dtype=torch.float
        )
        ic(m9_2)
        ic(m9_2.lift(u, v, d).shape)
        
        m10 = PinholeCameraHWFxyRatio(
            fxy_r = torch.ones([*prefix,2], device=device, dtype=torch.float), 
            H=100.,
            W=100., 
            device=device, dtype=torch.float
        )
        ic(m10)
        ic(m10.lift(u, v, d).shape)
        
        m11 = PinholeCameraHWF(
            f=torch.randn([*prefix], device=device, dtype=torch.float), 
            H=torch.randn([*prefix], device=device, dtype=torch.float), 
            W=torch.randn([*prefix], device=device, dtype=torch.float)
        )
        ic(m11)
        ic(m11.lift(u, v, d).shape)
    
    def test_getset(device=torch.device('cuda')):
        from icecream import ic
        prefix = [7,13]
        # prefix = ()
        
        m1 = PinholeCameraHWF(
            f=Scalar(torch.full(prefix, 200.0, device=device, dtype=torch.float)), 
            H=Scalar(torch.full(prefix, 200.0, device=device, dtype=torch.float)), 
            W=Scalar(torch.full(prefix, 200.0, device=device, dtype=torch.float)))
        ic(m1)
        t1 = m1[5]
        t1[5] = PinholeCameraHWF(f=100., H=100., W=100., device=device) # NOTE: the original m1 is also modified
        
        m2 = m1.new([9,5])
        m2[1:8] = m1[:, 3:8]

    test()
    test_nested()
    test_focal()
    test_stack()
    test_refine()
    test_repr()
    test_batched()
    test_getset()