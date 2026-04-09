import numpy as np
import os

basic_color = [
    [0.98, 0.02, 0.02], # red
               [0.02, 0.98, 0.02], # green
               [0.02, 0.02, 0.98], # blue
               [0.98, 0.98, 0.02], # yellow
               [0.98, 0.02, 0.98], # purple
               [0.02, 0.98, 0.98], # cyan
               [0.40, 0.02, 0.02], # dark red
               [0.98, 0.50, 0.02], # gold
               [0.54, 0.16, 0.82], # voilet
               [0.10, 0.10, 0.60], # navi blue
               [0.02, 0.02, 0.02], # black
               [0.3, 0.3, 0.3], # gray
               [0.7, 0.7, 0.7], # white
               ]
color_dist = 0.01

def get_draw_box_obj(bin_size, colorid=0):
    import open3d as o3d

    bx = bin_size[0]
    by = bin_size[1]
    bz = bin_size[2]
    points = [[0, 0, 0], [bx, 0, 0], [0, by, 0], [bx, by, 0], [0, 0, bz], [bx, 0, bz],
              [0, by, bz], [bx, by, bz]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [basic_color[colorid] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    text = "BIN:%dx%dx%d" % (bx, by, bz)
    # tt = o3d.t.geometry.TriangleMesh.create_text(text, depth=1)
    # tt.paint_uniform_color((1,0,0))
    ss = 0.4
    tt = None
    # tt.scale(ss, (0, 0, 0))
    # tt.translate((0, by, bz))
    return line_set, tt

def render_all(boxes, bin_size, colors, save_figure=False, output_image_path = None, step_counter = None):
    # return

    import open3d as o3d
    import open3d.visualization as vis
    if save_figure:
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)

    draw_obj = []
    assert  bin_size is not None
    bin_size = np.array(bin_size)
    bin_obj, text_obj = get_draw_box_obj(bin_size, -1)
    draw_obj.append(bin_obj)
    # draw_obj.append(text_obj)
    cubes = []
    for id, pbox in enumerate(boxes):
        ob = o3d.geometry.TriangleMesh.create_box(pbox[0], pbox[1], pbox[2])
        ob.compute_vertex_normals()

        if id < len(colors):
            color = colors[id]
        else:
            color = 1 - np.random.uniform(0.0, 1.0, size=[1, 3])
            colors.append(color)

        ob.paint_uniform_color(color[0])
        ob.translate((pbox[3], pbox[4], pbox[5]))
        draw_obj.append(ob)
        cubes.append(ob)
    if step_counter is None:
        step_counter = len(boxes)
    if not save_figure:
        vis.draw(draw_obj)
    else:
        scene = o3d.visualization.rendering.OffscreenRenderer(640, 480)
        for i, cube in enumerate(draw_obj):
            scene.scene.add_geometry(f"ele_{i}", cube, o3d.visualization.rendering.MaterialRecord())

        ratio = 0.8
        view_counter = 0
        view_num = 1
        for m in [1, -1]:
            for n in [1, -1]:
                camera_position = bin_size / 2 + np.array([m * ratio, n * ratio, ratio]) * bin_size
                look_at_point = bin_size / 2
                up_vector = [0, 0, 1]
                scene.scene.camera.look_at(look_at_point, camera_position, up_vector)
                image = scene.render_to_image()
                o3d.io.write_image(os.path.join(output_image_path, f'{step_counter}_{len(boxes) - 1}.png'), image)
                view_counter += 1
                if view_counter >= view_num:
                    break
            if view_counter >= view_num:
                break
        del scene
    del draw_obj,
    return colors
