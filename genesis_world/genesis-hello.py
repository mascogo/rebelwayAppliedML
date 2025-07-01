import genesis as gs
gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=True)
box = scene.add_entity(gs.morphs.Box())
# plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file=r'C:\Users\usuario\anaconda3\envs\genesis\Lib\site-packages\genesis\assets\xml\franka_emika_panda\panda.xml'),
)

scene.build()

for i in range(1000):
    scene.step()