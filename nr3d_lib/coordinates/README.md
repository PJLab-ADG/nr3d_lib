# Coordinates
An ongoing list of various 3D coordinate systems.

## OpenCV / COLMAP / Open3D / standard pinhole camera in textbook
- right-handed
- [open3d](https://github.com/isl-org/Open3D/issues/1347#issuecomment-558205561)
```python
"""
< opencv / colmap convention >                 
Facing [+z] direction, y downwards, x right,     
                 z                                 
               ↗                                   
              /                                    
             /                                     
            o------> x                             
            |                                      
            |
            |
            ↓ 
            y
"""
```

## OpenGL / Blender

```python
"""
< openGL / blender convention >
Facing [-z] direction, y upwards, x right
            y
            ↑
            |
            |
            |
            o-------> x
           / 
          /  
         ↙   
        z    
"""
```

## Unreal Engine / CARLA
- :warning: left-handed :warning: 
- [source](https://carla.readthedocs.io/en/latest/python_api/#carlarotation)
  - > CARLA uses the Unreal Engine coordinates system. This is a Z-up left-handed system.

```python
"""
< carla / UE convention >
Facing [+x] direction, z upwards, y right
        z ↑ 
          |  ↗ x
          | /
          |/
          o--------->
                    y
            
            
             
            
"""
```

## Unity
- :warning: left-handed :warning: 
- 

```python
"""
< Unity convention >
Facing [+z] direction, y upwards, x right
        y ↑ 
          |  ↗ z
          | /
          |/
          o--------->
                    x
            
            
             
            
"""
```

## VTK / mayavi

- [source](https://kitware.github.io/vtk-examples/site/VTKBook/08Chapter8/#81-coordinate-systems)
```python
"""
< VTK convention >
Facing [+y] direction, z upwards, x right
        z ↑ 
          |  ↗ y
          | /
          |/
          o--------->
                    x
            
            
             
            
"""
```

## ROS / waymo
- right-handed
- [source](https://waymo.com/open/data/perception/), in #Coordinate Systems
  - > The x-axis points down the lens barrel out of the lens. The z-axis points up. The y/z plane is parallel to the camera plane. The coordinate system is right handed.
```python
"""
< ROS / waymo convention >
Facing [+x] direction, z upwards, y left
        z ↑ 
          |  ↗ x
          | /
          |/
 <--------o
y
            
            
             
            
"""
```


## habitat_sim

- [ ] check habitat_sim

```python
"""
 < habitat_sim convention >
Facing [+x] direction, y upwards, z right
        y ↑ 
          |  ↗ x
          | /
          |/
          o--------->
                    z
"""
```
