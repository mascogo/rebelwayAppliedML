import hou
import math
import sys
import os
from pprint import pformat

sys.path.append(os.path.join(os.environ.get("HIP"), "scripts"))

from a_star import AStarPathFinding

    
    
def get_maze_from_grid():
    grid = hou.pwd().parm('grid_path').eval()
    grid_geo = hou.node(grid).geometry()
    
    prims = grid_geo.prims()
    
    num_rows = hou.pwd().parm('rows').eval()
    num_cols = hou.pwd().parm('cols').eval()
    
    grid_matrix = []
    
    for row_idx in range(num_rows):
        new_row = []
        for col_idx in range(num_cols):
            prim_idx = row_idx * num_cols + col_idx
            prim = grid_geo.prim(prim_idx)
            color = prim.attribValue("Cd")
            new_row.append(1 if color == (1.0, 1.0, 1.0) else 0)
        grid_matrix.append(new_row) 
        
    return grid_matrix
    
    
def grid_position_object(obj_path): 
    grid_path = hou.pwd().parm('grid_path').eval()
    grid_node = hou.node(grid_path)
    rows = hou.pwd().parm('rows').eval()
    cols = hou.pwd().parm('cols').eval()
    node = hou.node(obj_path)
    world_position = node.parmTuple("t").eval()
    world_x = round(world_position[0])
    world_z = round(world_position[2])
    
    offset_col = 1 if (cols % 2) == 0 else 0
    offset_row = 1 if (rows % 2) == 0 else 0
    col = world_x + cols//2 - offset_col
    row = world_z + rows //2 - offset_row
    
    # main_char.parmTuple("t").set((world_x, 0, world_z))
    
    pos = (row, col)
    print("world: {} --> grid: {}".format((world_x, world_z), pos))
    return pos

def grid2world_pos(pos):
    row, col = pos
    rows = hou.pwd().parm('rows').eval()
    cols = hou.pwd().parm('cols').eval()
    
    x_min = cols//2 - cols
    z_min = rows//2 - rows
    x_pos = x_min + col + 1
    z_pos = z_min + row + 1
    return (x_pos, z_pos)
    

def delete_animation(obj_path):
    obj = hou.node(obj_path)
    for p in obj.parms():
        p.deleteAllKeyframes()

        
def anim_npc(npc_path, path, factor = 1):
    delete_animation(npc_path)
    npc = hou.node(npc_path)
    
    tx = npc.parm("tx")
    tz = npc.parm("tz")
    
    tx.setExpression("linear()")
    tz.setExpression("linear()")
    for i, g_pos in enumerate(path):
        frame = i*factor + 1
        w_pos = grid2world_pos(g_pos)
        print("{} -> {}".format(g_pos, w_pos))
        x, z = w_pos
        new_key_x = hou.Keyframe()
        new_key_x.setValue(x)
        new_key_x.setFrame(frame)

        new_key_z = hou.Keyframe()
        new_key_z.setValue(z)
        new_key_z.setFrame(frame)
        
        tx.setKeyframe(new_key_x)
        tz.setKeyframe(new_key_z)

        
def solve_maze():

    main_char_path = hou.pwd().parm('main_char_path').eval()
    npc_paths = []    
    
    target_grid_pos = grid_position_object(main_char_path)
    tx = hou.node(main_char_path).parm('tx')
    tz = hou.node(main_char_path).parm('tz')
    fixed_pos = grid2world_pos(target_grid_pos) 
    tx.set(fixed_pos[0])
    tz.set(fixed_pos[1])
    
    maze1 = get_maze_from_grid()
    print("")
    for row in maze1:
        print(row)
    
    print("target_grid_pos: {}".format(target_grid_pos))
    for n in range(hou.pwd().parm('npcs').eval()):
        npc_parm = 'npc_{}'.format(n+1)
        print("npc_parm: '{}'".format(npc_parm))
        npc_path = hou.pwd().parm(npc_parm).eval()
        if npc_path and hou.node(npc_path):
            npc_paths.append(npc_path)
    paths = []
    for npc_path in npc_paths:
        delete_animation(npc_path)
        npc_world_pos = hou.node(npc_path).parmTuple('t').eval() 
        print("{}: {}".format(npc_path, npc_world_pos))
        npc_grid_pos = grid_position_object(npc_path)
        print("----> {}".format(npc_grid_pos))
        
        # if npc ot of the grid, can't move
        if npc_grid_pos[0] < 0 or npc_grid_pos[0] > len(maze1) or npc_grid_pos[1] < 0 or npc_grid_pos[1] > len(maze1[0]):
            continue
                
        # if npc in a wall, can't move
        if maze1[npc_grid_pos[0]][npc_grid_pos[1]] == 0:
            continue
            
        pathfinder = AStarPathFinding(maze1, npc_grid_pos, target_grid_pos)
        path = pathfinder.find_path()
        print("path: {}".format(path))
        
        # clean npc former animation
        
        
        
        if path:
            anim_npc(npc_path, path, hou.pwd().parm('anim_stretch_factor').eval())

            paths.append(path)
            
    if not paths:
        print("can't find any path!")
          
                
                
        # hou.node(npc_path).parmTuple('t').set((round(world_pos[0]), round(world_pos[1]), round(world_pos[2])))
    
        
        
        
def run_this():
    solve_maze()
