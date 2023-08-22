"""
@file   conversion.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Tutorials on conversion among coordinate systems
"""

import numpy as np

"""
NOTE: The following assumes applying left-multiplications to matrices or 3x1/4x1 column vectors.
NOTE: '@' signifies matrix multiplication.

A quick reminder: How coordinate conversions are passed on:

    vectors_in_bbb = aaa_to_bbb  @  vectors_in_aaa
    
    vectors_in_ccc = bbb_to_ccc  @  aaa_to_bbb  @  vectors_in_aaa
                                    <----------------------------
                                    This converts vectors in aaa to vectors in bbb.
                    <--------------
                    This converts vectors in bbb to vectors in ccc.
"""



"""
Tutorial on how to derive conversions between vectors in different coordinates:
    
    Say you want to convert vectors in carla's camera to vectors in opencv's camera:
    
     < carla / UE convention >                >>>>  < opencv / colmap convention >                 
    facing [+x] direction, z upwards, y right       facing [+z] direction, y downwards, x right,     
        z ↑                                                      z'                            
          |  ↗ x                                               ↗                                   
          | /                                                 /                                    
          |/                                                 /                                     
          o--------->                                       o------> x'                          
                    y                                       |                                      
                                                            |
                                                            |
                                                            ↓ 
                                                            y'
    
    ∵   x' <- y:        we want original y vectors to become x' vectors.
    ∵   y' <- -z:       we want original -z vectors to become y' vectors.
    ∵   z' <- x:        we want original x vectors to become z' vectors.
    
    ∴ we have:
        [ x' ]   [ 0  1  0]   [ x ]
        [ y' ] = [ 0  0 -1] @ [ y ]
        [ z' ]   [ 1  0  0]   [ z ]
    
    also:
        vecs_in_opencv = carla_to_opencv  @  vecs_in_carla
    
    hence:
                            [ 0  1  0]
        carla_to_opencv  =  [ 0  0 -1]
                            [ 1  0  0]
"""



""" 
Tutorial on How to derive conversions between camera_to_world matrices in different coordinates:

    Say you have a cameraxxx_to_world matrix that specifies a camera follows xxx coordinate convention;
        and you want to convert it into yyy coordinate convention;
    
    Then:
    
        camerayyy_to_world = cameraxxx_to_world  @  yyy_to_xxx
        
    Explanation:
    
        vecs_in_xxxworld  =  cameraxxx_to_world  @  yyy_to_xxx  @  vecs_in_camerayyy
                                                <------------------------------
                                                this converts vectors in yyy camera to xxx camera
                            <--------------------
                            this converts vectors in xxx camera to world
                        
        vecs_in_xxxworld  =  [[[     camerayyy_to_world    ]]]  @  vecs_in_camerayyy
    
    e.g. to convert a waymo's c2w matrix to a opencv's c2w matrix:
        camera_opencv_to_world  =  camera_waymo_to_world  @  opencv_to_waymo
        
        Explanation:
            vecs_in_waymo_world = camera_waymo_to_world  @  opencv_to_waymo  @  vecs_in_camera_opencv
            vecs_in_waymo_world = [[[       camera_opencv_to_world      ]]]  @  vecs_in_camera_opencv

"""

opencv_to_waymo = np.array(
    [[0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]])

opencv_to_carla = np.array(
    [[0, 0, 1],
     [1, 0, 0],
     [0,-1, 0]])

carla_to_opencv = np.array(
    [[0, 1, 0],
     [0, 0, -1],
     [1, 0, 0]])