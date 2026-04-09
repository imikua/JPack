import numpy as np
from functools import reduce
from .convex_hull import ConvexHull, point_in_polygen
from .PctTools import AddNewEMSZ, maintainEventBottom

from copy import deepcopy

tolerance = 1e-6
item_decimals = 6

from time import time as clock

class Stack(object):
    def __init__(self, centre, mass):
        self.centre = centre
        self.mass = mass

class DownEdge(object):
    def __init__(self, box):
        self.box = box
        self.area = None
        self.centre2D = None

class EMSNode(object):
    def __init__(self, parent, node_vec, size_minimum):
        self.node_vec = node_vec
        self.lx, self.ly, self.lz = self.node_vec[0:3]
        self.x,  self.y,  self.z    = np.around(self.node_vec[3:6] - self.node_vec[0:3], item_decimals)
        self.size_minimum = size_minimum

        self.parent = parent
        self.children = []
        self.placed_items = []          # 当前 ems 被摆放的物体 (x1, y1, z1, x2, y2, z2), real size
        self.valid_children = 0         # 有效子节点数
        self.is_valid = True            # 是否被其他 leaf EMS 包含
        self.internal_nodes = []        # internal nodes
        self.leaf_nodes = []            # 有效的摆放位置

        self.V = 1                      # V 取相对值
        self.volume = 0                 # volume 取 real value
        self.td_target = self.volume + self.V
        self.sub_container = False      # 当前 ems 是否能够成为 sub-container
        # self.tree_node_num = 1          # 当前子树的节点数


    def _isUsableEMS(self, node_vec):
        x1, y1, z1, x2, y2, z2 = node_vec
        xd = x2 - x1 + tolerance
        yd = y2 - y1 + tolerance
        zd = z2 - z1 + tolerance
        if (xd >= self.size_minimum) and (yd >= self.size_minimum)\
            and (zd >= self.size_minimum):
            return True
        return False

    def checkSubContainer(self, box_bound):
        # return True
        if len(self.placed_items) > box_bound:
            self.sub_container = False
        else:
            self.sub_container = True
            if len(self.placed_items) and not self.valid_children:
                self.sub_container = False
        return self.sub_container

    def addChild(self, next_vec):
        if self._isUsableEMS(next_vec):
            child = EMSNode(self, next_vec, self.size_minimum)
            self.children.append(child)
            self.valid_children += 1

    # todo: maybe we can rewrite this part and use batch calculation to speed up
    def updateNode(self, itemLocation, new_node_num):
        # V is get directly in PCT_policy, so not updated here
        # only stores item that intersects with the ems
        # self.tree_node_num += new_node_num
        if itemLocation not in self.placed_items:
            self.placed_items.append(itemLocation)

        if self.parent is not None:
            self.parent.updateNode(itemLocation, new_node_num)

    def disable(self):
        # self.tree_node_num -= 1
        # if not len(self.placed_items):
            # self.is_valid = False
            # self.parent.children.remove(self)
            # self.parent.valid_children -= 1
        # if self.parent is None:
        #     return
        # self.parent.disable()
        self.is_valid = False
        self.parent.children.remove(self)
        self.parent.valid_children -= 1

    def getInternalNodes(self):
        if len(self.placed_items):
            node_vec = np.array(self.node_vec, dtype=float)
            placed_items = np.array(self.placed_items, dtype=float)
            self.internal_nodes = np.concatenate((-1 * np.ones(3), np.ones(3)), axis=0) *  \
                np.around(np.minimum(
                np.concatenate((-1 * np.ones(3), np.ones(3)), axis=0) * node_vec, 
                np.concatenate((-1 * np.ones(3), np.ones(3)), axis=0) * placed_items
            ), decimals=item_decimals)

            # calculate volumes
            self.volume = np.sum(np.prod(self.internal_nodes[:, 3:6] - self.internal_nodes[:, 0:3], axis=1))
        else:
            self.internal_nodes = np.array(self.placed_items, dtype=float)
            self.volume = 0

        return self.internal_nodes

    def getScaledInternalNodes(self):
        maxDim = max([self.x, self.y, self.z])

        if len(self.placed_items):
            node_vec = np.array(self.node_vec, dtype=float)
            placed_items = np.array(self.placed_items, dtype=float)
            self.internal_nodes = np.concatenate((-1 * np.ones(3), np.ones(3)), axis=0) *  \
                np.around(np.minimum(
                np.concatenate((-1 * np.ones(3), np.ones(3)), axis=0) * node_vec,
                np.concatenate((-1 * np.ones(3), np.ones(3)), axis=0) * placed_items
            ), decimals=item_decimals)
            # calculate volumes
            self.volume = np.sum(np.prod(self.internal_nodes[:, 3:6] - self.internal_nodes[:, 0:3], axis=1))
        else:
            self.internal_nodes = np.array(self.placed_items, dtype=float)
            self.volume = 0
        expand = False
        if self.x < maxDim:
            expand = True
            x_expansion = np.round([self.lx + self.x, self.ly, self.lz, self.lx + maxDim, self.ly + self.y, self.lz + maxDim], item_decimals).reshape(1, -1)
            self.internal_nodes = np.concatenate([self.internal_nodes, x_expansion], axis=0)

        if self.y < maxDim:
            y_expansion = np.round([self.lx, self.ly + self.y, self.lz, self.lx + self.x, self.ly + maxDim, self.lz + maxDim], item_decimals).reshape(1, -1)
            self.internal_nodes = np.concatenate([self.internal_nodes, y_expansion], axis=0)

            if expand:
                xy_expansion = np.round([self.lx + self.x, self.ly + self.y, self.lz, self.lx + maxDim, self.ly + maxDim, self.lz + maxDim], item_decimals).reshape(1, -1)
                self.internal_nodes = np.concatenate([self.internal_nodes, xy_expansion], axis=0)
        # location = np.array([self.lx, self.ly, self.lz])
        # print(self.internal_nodes.shape)
        # print(location.shape)
        # if len(self.internal_nodes) != 0:
        #     if np.min(np.round(self.internal_nodes.reshape(-1,3) - location, item_decimals)) < 0:
        #         print(np.round(self.internal_nodes.reshape(-1,3) - location, item_decimals))
        #         print(self.internal_nodes)
        #         assert False
        return self.internal_nodes

    # def checkOverlap(self):
    #     overlap_objs = 0
    #     for i in range(len(self.internal_nodes)):
    #         item = np.array(self.internal_nodes[i])
    #         item[:3] = item[:3] * -1
    #         for j in range(i+1, len(self.internal_nodes)):
    #             box = np.array(self.internal_nodes[j])
    #             box[:3] = box[:3] * -1
    #             print(box, item)
    #             intersect = np.around(np.minimum(box, item), 6)
    #             signal3D = (intersect[0] + intersect[3] > 0) * (intersect[1] + intersect[4] > 0) * (intersect[2] + intersect[5] > 0)
    #             index3D = np.where(signal3D.squeeze())[0]
    #             if len(index3D):
    #                 item[:3] = item[:3] * -1
    #                 box[:3] = box[:3] * -1
    #                 print(item)
    #                 print(box)
    #                 print()
    #                 overlap_objs += 1
    #     if overlap_objs:
    #         print("overlap objs: ", overlap_objs)
    #         return True
    #     return False


class Box(object):
    def __init__(self, x, y, z, lx, ly, lz, density, virtual=False):
        self.x, self.y, self.z, self.lx, self.ly, self.lz = np.around([x, y, z, lx, ly, lz], item_decimals)
        self.centre = np.array([self.lx + self.x / 2, self.ly + self.y / 2, self.lz + self.z / 2]).round(item_decimals)
        self.vertex_low = np.array([self.lx, self.ly, self.lz])
        self.vertex_high = np.array([self.lx + self.x, self.ly + self.y, self.lz + self.z]).round(item_decimals)
        self.mass = x * y * z * density
        if virtual: self.mass *= 1.0
        self.bottom_edges = []
        self.bottom_whole_contact_area = None

        self.up_edges = {}
        self.up_virtual_edges = {}

        self.thisStack = Stack(self.centre, self.mass)
        self.thisVirtualStack = Stack(self.centre, self.mass)
        self.involved = False


    def calculate_new_com(self, virtual=False):
        new_stack_centre = self.centre * self.mass
        new_stack_mass = self.mass

        for ue in self.up_edges.keys():
            if not ue.involved:
                new_stack_centre += self.up_edges[ue].centre * self.up_edges[ue].mass
                new_stack_mass += self.up_edges[ue].mass

        for ue in self.up_virtual_edges.keys():
            if ue.involved:
                new_stack_centre += self.up_virtual_edges[ue].centre * self.up_virtual_edges[ue].mass
                new_stack_mass += self.up_virtual_edges[ue].mass

        new_stack_centre /= new_stack_mass
        if virtual:
            self.thisVirtualStack.mass = new_stack_mass
            self.thisVirtualStack.centre = new_stack_centre
        else:
            self.thisStack.mass = new_stack_mass
            self.thisStack.centre = new_stack_centre

    def calculated_impact(self):
        if len(self.bottom_edges) == 0:
            return True
        elif not point_in_polygen(self.thisStack.centre[0:2],
                                  self.bottom_whole_contact_area):
            return False
        else:
            if len(self.bottom_edges) == 1:
                stack = self.thisStack
                self.bottom_edges[0].box.up_edges[self] = stack
                self.bottom_edges[0].box.calculate_new_com()
                if not self.bottom_edges[0].box.calculated_impact():
                    return False
            else:
                direct_edge = None
                for e in self.bottom_edges:
                    if self.thisStack.centre[0] - e.area[0] > 1e-6 and e.area[2] - self.thisStack.centre[0] > 1e-6  \
                            and self.thisStack.centre[1] - e.area[1] > 1e-6 and e.area[3] - self.thisStack.centre[1] > 1e-6:
                        direct_edge = e
                        break

                if direct_edge is not None:
                    for edge in self.bottom_edges:
                        if edge == direct_edge:
                            edge.box.up_edges[self] = self.thisStack
                            edge.box.calculate_new_com()
                        else:
                            edge.box.up_edges[self] = Stack(self.thisStack.centre, 0)
                            edge.box.calculate_new_com()

                    for edge in self.bottom_edges:
                        if not edge.box.calculated_impact():
                            return False

                elif len(self.bottom_edges) == 2:
                    com2D = self.thisStack.centre[0:2]

                    tri_base_line = self.bottom_edges[0].centre2D - self.bottom_edges[1].centre2D
                    tri_base_len = np.linalg.norm(tri_base_line)
                    tri_base_line /= tri_base_len ** 2

                    ratio0 = abs(np.dot(com2D - self.bottom_edges[1].centre2D, tri_base_line))
                    ratio1 = abs(np.dot(com2D - self.bottom_edges[0].centre2D, tri_base_line))

                    com0 = np.array([*self.bottom_edges[0].centre2D, self.thisStack.centre[2]])
                    com1 = np.array([*self.bottom_edges[1].centre2D, self.thisStack.centre[2]])

                    stack0 = Stack(com0, self.thisStack.mass * ratio0)
                    stack1 = Stack(com1, self.thisStack.mass * ratio1)

                    self.bottom_edges[0].box.up_edges[self] = stack0
                    self.bottom_edges[0].box.calculate_new_com()

                    self.bottom_edges[1].box.up_edges[self] = stack1
                    self.bottom_edges[1].box.calculate_new_com()

                    if not self.bottom_edges[0].box.calculated_impact():
                        return False
                    if not self.bottom_edges[1].box.calculated_impact():
                        return False

                else:
                    com2D = self.thisStack.centre[0:2]
                    length = len(self.bottom_edges)
                    coefficient = np.zeros((int(length * (length - 1) / 2 + 1), length))
                    value = np.zeros((int(length * (length - 1) / 2 + 1), 1))
                    counter = 0
                    for i in range(length - 1):
                        for j in range(i + 1, length):
                            tri_base_line = self.bottom_edges[i].centre2D - self.bottom_edges[j].centre2D
                            molecular = np.dot(com2D - self.bottom_edges[i].centre2D, tri_base_line)
                            if molecular != 0:
                                ratioI2J = abs(np.dot(com2D - self.bottom_edges[j].centre2D, tri_base_line)) / molecular
                                coefficient[counter, i] = 1
                                coefficient[counter, j] = - ratioI2J
                            counter += 1

                    coefficient[-1, :] = 1
                    value[-1, 0] = 1
                    assgin_ratio = np.linalg.lstsq(coefficient, value, rcond=None)[0]

                    for i in range(length):
                        e = self.bottom_edges[i]
                        newAdded_mass = self.thisStack.mass * assgin_ratio[i][0]
                        newAdded_com = np.array([*e.centre2D, self.thisStack.centre[2]])
                        e.box.up_edges[self] = Stack(newAdded_com, newAdded_mass)
                        e.box.calculate_new_com()

                    for e in self.bottom_edges:
                        if not e.box.calculated_impact():
                            return False
            return True

    def calculated_impact_virtual(self, first=False):
        self.involved = True
        if len(self.bottom_edges) == 0:
            self.involved = False
            return True
        elif not point_in_polygen(self.thisVirtualStack.centre[0:2],
                                  self.bottom_whole_contact_area):
            self.involved = False
            return False
        else:
            if len(self.bottom_edges) == 1:
                stack = self.thisVirtualStack
                self.bottom_edges[0].box.up_virtual_edges[self] = stack
                self.bottom_edges[0].box.calculate_new_com(True)
                if not self.bottom_edges[0].box.calculated_impact_virtual():
                    self.involved = False
                    return False
            else:
                direct_edge = None
                for e in self.bottom_edges:
                    if self.thisVirtualStack.centre[0] - e.area[0] > 1e-6 and e.area[2] - self.thisVirtualStack.centre[0] > 1e-6  \
                            and self.thisVirtualStack.centre[1] - e.area[1] > 1e-6 and e.area[3] - self.thisVirtualStack.centre[1] > 1e-6 :
                        direct_edge = e
                        break

                if direct_edge is not None:
                    for edge in self.bottom_edges:
                        if edge == direct_edge:
                            edge.box.up_virtual_edges[self] = self.thisVirtualStack
                            edge.box.calculate_new_com(True)
                        else:
                            edge.box.up_virtual_edges[self] = Stack(self.centre, 0)
                            edge.box.calculate_new_com(True)

                    for edge in self.bottom_edges:
                        if not edge.box.calculated_impact_virtual():
                            self.involved = False
                            return False

                elif len(self.bottom_edges) == 2:
                    com2D = self.thisVirtualStack.centre[0:2]

                    tri_base_line = self.bottom_edges[0].centre2D - self.bottom_edges[1].centre2D
                    tri_base_len = np.linalg.norm(tri_base_line)
                    tri_base_line /= tri_base_len ** 2

                    ratio0 = abs(np.dot(com2D - self.bottom_edges[1].centre2D, tri_base_line))
                    ratio1 = abs(np.dot(com2D - self.bottom_edges[0].centre2D, tri_base_line))

                    com0 = np.array([*self.bottom_edges[0].centre2D, self.thisVirtualStack.centre[2]])
                    com1 = np.array([*self.bottom_edges[1].centre2D, self.thisVirtualStack.centre[2]])

                    stack0 = Stack(com0, self.thisVirtualStack.mass * ratio0)
                    stack1 = Stack(com1, self.thisVirtualStack.mass * ratio1)

                    self.bottom_edges[0].box.up_virtual_edges[self] = stack0
                    self.bottom_edges[0].box.calculate_new_com(True)
                    self.bottom_edges[1].box.up_virtual_edges[self] = stack1
                    self.bottom_edges[1].box.calculate_new_com(True)

                    if not self.bottom_edges[0].box.calculated_impact_virtual() \
                            or not self.bottom_edges[1].box.calculated_impact_virtual():
                        self.involved = False
                        return False

                else:
                    com2D = self.thisVirtualStack.centre[0:2]
                    length = len(self.bottom_edges)
                    coefficient = np.zeros((int(length * (length - 1) / 2 + 1), length))
                    value = np.zeros((int(length * (length - 1) / 2 + 1), 1))
                    counter = 0
                    for i in range(length - 1):
                        for j in range(i + 1, length):
                            tri_base_line = self.bottom_edges[i].centre2D - self.bottom_edges[j].centre2D
                            molecular = np.dot(com2D - self.bottom_edges[i].centre2D, tri_base_line)
                            if molecular != 0:
                                ratioI2J = abs(np.dot(com2D - self.bottom_edges[j].centre2D, tri_base_line)) / molecular
                                coefficient[counter, i] = 1
                                coefficient[counter, j] = -ratioI2J
                            counter += 1

                    coefficient[-1, :] = 1
                    value[-1, 0] = 1
                    x = np.linalg.lstsq(coefficient, value, rcond=None)
                    assgin_ratio = x[0]
                    for i in range(length):
                        e = self.bottom_edges[i]
                        newAdded_mass = self.thisVirtualStack.mass * assgin_ratio[i][0]
                        newAdded_com = np.array([*e.centre2D, self.thisVirtualStack.centre[2]])
                        e.box.up_virtual_edges[self] = Stack(newAdded_com, newAdded_mass)
                        e.box.calculate_new_com(True)

                    for e in self.bottom_edges:
                        if not e.box.calculated_impact_virtual():
                            self.involved = False
                            return False

            if first:
                for e in self.bottom_edges:
                    e.box.up_virtual_edges.pop(self)
            self.involved = False
            return True

class Space(object):
    def __init__(self, width=10, length=10, height=10, size_minimum=0, holder = 60, box_bound=30, args = None):
        self.plain_size = np.array([width, length, height])
        self.max_axis = max(width, length)
        self.height = height
        self.low_bound = size_minimum
        self.box_bound = box_bound

        self.upLetter = []
        self.box_vec  = []
        self.reset()
        self.args = args
        self.large_scale = args.large_scale
        self.update_container_method = args.update_container_method

    def reset(self):
        self.upLetter = []
        self.letterIdx = 0

        self.box_vec = []
        self.box_vec.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

        self.root_ems = EMSNode(None, np.array([0, 0, 0, *self.plain_size]), self.low_bound)
        self.root_ems.sub_container = True
        self.leaf_emss = [self.root_ems]

        self.sub_containers = set([self.root_ems])        # store the chosen sub-containers
        self.all_itemLocations = []                       # store all the item locations

        self.boxes = []
        self.box_idx = 0
        self.serial_number = 0

        self.ZMAP = dict()
        self.ZMAP[0] = dict()

        r = self.ZMAP[0]
        r['x_up'] = [0]
        r['y_left'] = [0]
        r['x_bottom'] = [self.plain_size[0]]
        r['y_right'] = [self.plain_size[1]]

    def interSect2D(self, box):
        if self.box_idx == 0:
            return 0, [], []
        upLetter = np.array(self.upLetter[0: self.box_idx])
        intersect = np.around(np.minimum(box, upLetter), 6)
        signal = (intersect[:, 0] + intersect[:, 2] > 0) * (intersect[:, 1] + intersect[:, 3] > 0) # 等于零的地方表示不相交
        index = np.where(signal)[0]
        if len(index) == 0:
            return 0, [], []
        else:
            return np.max(upLetter[index, 4]), index, intersect[index]

    def get_ratio(self):
        vo = reduce(lambda x, y: x + y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        mx = self.plain_size[0] * self.plain_size[1] * self.plain_size[2]
        ratio = vo / mx
        if not ratio - 1e-6 <= 1.0:
            print("ratio error, the ratio is: ", ratio)
            for box in self.boxes:
                print(box.x, box.y, box.z, box.lx, box.ly, box.lz)
        assert ratio - 1e-6 <= 1.0
        return ratio

    def scale_down(self, bottom_whole_contact_area):
        centre2D = np.mean(bottom_whole_contact_area, axis=0)
        dirction2D = bottom_whole_contact_area - centre2D
        bottom_whole_contact_area -= dirction2D * 0.1
        return bottom_whole_contact_area.tolist()

    def drop_box(self, box_size, idx, flag, density, setting, **kwags):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx

        if lx + x - 1e-6 > self.plain_size[0] or ly + y - 1e-6 > self.plain_size[1]:
            return False
        if lx + 1e-6 < 0 or ly + 1e-6 < 0:
            return False
        box_info = np.array([-lx, -ly, lx + x, ly + y, 0])
        max_h, interIdx, interArea = self.interSect2D(box_info)
        if max_h + z - 1e-6 > self.height:
            return False
        box_info[-1] = max_h + z
        box_now = Box(x, y, z, lx, ly, max_h, density)

        if setting != 2:
            combine_contact_points = []
            for inner in range(len(interIdx)):
                idx = interIdx[inner]
                tmp = self.boxes[idx]
                if abs(tmp.lz + tmp.z - max_h) < tolerance:
                    x1, y1, x2, y2, _ = interArea[inner]
                    x1, y1 = -x1, -y1
                    newEdge = DownEdge(tmp)
                    newEdge.area = (x1, y1, x2, y2)
                    newEdge.centre2D = np.array([x1 + x2, y1 + y2]) / 2
                    box_now.bottom_edges.append(newEdge)
                    combine_contact_points.append([x1, y1])
                    combine_contact_points.append([x1, y2])
                    combine_contact_points.append([x2, y1])
                    combine_contact_points.append([x2, y2])

            if len(combine_contact_points) > 0:
                box_now.bottom_whole_contact_area = self.scale_down(ConvexHull(combine_contact_points))
        sta_flag = self.check_box(max_h, box_now, setting)
        if sta_flag:
            self.boxes.append(box_now)  # record rotated box
            self.upLetter.append(box_info)  # record the box info -lx, -ly, lx + x, ly + y, max_h + z
            if self.box_idx == 0:
                self.box_vec =     [[lx, ly, max_h, lx + x, ly + y, max_h + z, 0, 0, 1]]
            else:
                self.box_vec.append([lx, ly, max_h, lx + x, ly + y, max_h + z, 0, 0, 1])
            self.box_idx += 1
            return True
        return False

    # Virtually place an item into the bin,
    # this function is used to check whether the placement is feasible for the current item
    def drop_box_virtual(self, box_size, idx, flag, density, setting, returnH = False, **kwargs):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx
        checkResult = True
        if lx + x - 1e-6 > self.plain_size[0] or ly + y - 1e-6 > self.plain_size[1]:
            checkResult = False
        if lx + 1e-6 < 0 or ly + 1e-6 < 0:
            checkResult = False

        box_info = np.array([-lx, -ly, lx + x, ly + y, 0])
        max_h, interIdx, interArea = self.interSect2D(box_info)

        if max_h + z - 1e-6 > self.height:
            checkResult = False

        box_now = Box(x, y, z, lx, ly, max_h, density, True)

        if setting != 2 and checkResult:
            combine_contact_points = []
            for inner in range(len(interIdx)):
                idx = interIdx[inner]
                tmp = self.boxes[idx]
                if abs(tmp.lz + tmp.z - max_h) < tolerance:
                    x1, y1, x2, y2, _ = interArea[inner]
                    x1, y1 = -x1, -y1

                    newEdge = DownEdge(tmp)
                    newEdge.area = (x1, y1, x2, y2)
                    newEdge.centre2D = np.array([x1 + x2, y1 + y2]) / 2
                    box_now.bottom_edges.append(newEdge)
                    combine_contact_points.append([x1, y1])
                    combine_contact_points.append([x1, y2])
                    combine_contact_points.append([x2, y1])
                    combine_contact_points.append([x2, y2])

            if len(combine_contact_points) > 0:
                box_now.bottom_whole_contact_area = self.scale_down(ConvexHull(combine_contact_points))

        if returnH:
            return checkResult and self.check_box(max_h, box_now, setting, True), max_h
        else:
            return checkResult and self.check_box(max_h, box_now, setting, True)

    # Check if the placement is feasible
    def check_box(self, max_h, box_now, setting, virtual=False):
        assert isinstance(setting, int), 'The environment setting should be integer.'
        if setting == 2:
            return True
        else:
            if abs(max_h) <  tolerance:
                return True
            if not virtual:
                result = box_now.calculated_impact()
                return result
            else:
                return box_now.calculated_impact_virtual(True)


    def interSectEMS3D(self, itemLocation, box_idx, NOEMS, EMS):
        itemLocation[0:3] *= -1

        EMS = EMS[0:NOEMS].copy()
        EMS[:, 0:3] *= -1

        if box_idx == 0:
            return 0, [], []

        intersect = np.around(np.minimum(itemLocation, EMS), 6)
        signal = (intersect[:, 0] + intersect[:, 3] > 0) * (intersect[:, 1] + intersect[:, 4] > 0) * (intersect[:, 2] + intersect[:, 5] > 0)
        delindex = np.where(signal)[0] 
        intersect = intersect[delindex]
        intersect[:, 0:3] *= -1
        return delindex, intersect

    # Calculate the incrementally generated empty maximal spaces during the packing.
    def GENEMS(self, itemLocation):
        self.all_itemLocations.append(itemLocation)
        numofemss = len(self.leaf_emss)
        EMS = np.array([x.node_vec for x in self.leaf_emss]).round(item_decimals)

        if len(EMS) == 0:
            return

        delflag, intersect = self.interSectEMS3D(np.array(itemLocation), self.box_idx, numofemss, EMS)

        for idx in range(len(delflag)):
            emsIdx = delflag[idx]
            inter = intersect[idx]
            self.Difference(emsIdx, inter, itemLocation)

        if len(delflag) != 0:
            numofemss = len(self.leaf_emss)
            self.leaf_emss = [self.leaf_emss[i] for i in range(numofemss) if i not in delflag]

        self.EliminateInscribedEMS()
        if self.large_scale:
            if self.update_container_method == 'recursive':
                self.updateSubContainers()
            else:
                self.updateSubContainersCheb(itemLocation)

        # maintain the event point by the way
        cx_min, cy_min, cz_min, cx_max, cy_max, cz_max = itemLocation
        # bottom
        if cz_min < self.plain_size[2]:
            bottomRecorder = self.ZMAP[round(cz_min, 6)]
            cbox2d = [cx_min, cy_min, cx_max, cy_max]
            maintainEventBottom(cbox2d, bottomRecorder['x_up'], bottomRecorder['y_left'], bottomRecorder['x_bottom'],
                                bottomRecorder['y_right'], self.plain_size)

        if cz_max < self.plain_size[2]:
            AddNewEMSZ(itemLocation, self)

    # Split an EMS when it intersects a placed item
    def Difference(self, emsID, intersection, itemLocation):
        ems_node = self.leaf_emss[emsID]
        x1, y1, z1, x2, y2, z2 = ems_node.node_vec
        x3, y3, z3, x4, y4, z4 = intersection

        self.AddNewEMS(ems_node, np.array([x1, y1, z1, x3, y2, z2]))
        self.AddNewEMS(ems_node, np.array([x4, y1, z1, x2, y2, z2]))
        self.AddNewEMS(ems_node, np.array([x1, y1, z1, x2, y3, z2]))
        self.AddNewEMS(ems_node, np.array([x1, y4, z1, x2, y2, z2]))
        self.AddNewEMS(ems_node, np.array([x1, y1, z4, x2, y2, z2]))
        itemLocation = tuple(itemLocation)
        ems_node.updateNode(itemLocation, ems_node.valid_children)
        if ems_node.valid_children:
            self.leaf_emss.extend(ems_node.children)

    def AddNewEMS(self, cur_node, next_vec):
        cur_node.addChild(next_vec)

    # Eliminate redundant ems
    def EliminateInscribedEMS(self):
        numofemss = len(self.leaf_emss)
        delflags = np.zeros(numofemss)
        for i in range(numofemss):
            for j in range(numofemss):
                if i == j:
                    continue
                if (self.leaf_emss[i].node_vec[0] >= self.leaf_emss[j].node_vec[0] and self.leaf_emss[i].node_vec[1] >= self.leaf_emss[j].node_vec[1]
                        and self.leaf_emss[i].node_vec[2] >= self.leaf_emss[j].node_vec[2] and self.leaf_emss[i].node_vec[3] <= self.leaf_emss[j].node_vec[3]
                        and self.leaf_emss[i].node_vec[4] <= self.leaf_emss[j].node_vec[4] and self.leaf_emss[i].node_vec[5] <= self.leaf_emss[j].node_vec[5]):
                    # EMS[i] 被 EMS[j] 完全包含
                    delflags[i] = 1
                    self.leaf_emss[i].disable()
                    break
        self.leaf_emss = [self.leaf_emss[i] for i in range(numofemss) if delflags[i] != 1]
        return len(self.leaf_emss)

    # todo to be check later
    @staticmethod
    def isancestorEMS(ems1, ems2):
        # if ems1 is ancestor node of ems2
        if ((ems1.lx - tolerance <= ems2.lx) and (ems1.ly - tolerance <= ems2.ly) and (ems1.lz - tolerance <= ems2.lz)
            and ((ems1.lx+ems1.x + tolerance) >= (ems2.lx+ems2.x)) and ((ems1.ly + ems1.y + tolerance) >= (ems2.ly + ems2.y))
            and ((ems1.lz + ems1.z + tolerance) >= (ems2.lz+ems2.z))):
            return True
        else:
            return False

    def checkInternalNodes(self, sub_container, itemLocation):
        sub_container.updateNode(itemLocation, 0)
        return True
    #     if itemLocation in sub_container.placed_items:
    #         return False
    #     cur_item = np.around(itemLocation, item_decimals)
    #     ems = deepcopy(sub_container.node_vec)
    #     ems[:3] = -1 * ems[:3]
    #     cur_item[:3] = -1 * cur_item[:3]
    #     intersect = np.minimum(cur_item, ems)
    #     signal3D = (intersect[0] + intersect[3] > 0) * (intersect[1] + intersect[4] > 0) * (intersect[2] + intersect[5] > 0)
    #     index3D = np.where(signal3D.squeeze())[0]
    #     if len(index3D): # intersect
    #         sub_container.updateNode(itemLocation, 0)
    #         return True
    #     else:
    #         return False

    @staticmethod
    def calArray2dDiff(array_0, array_1):
        if len(array_1) == 0: return array_0
        array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
        array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])
    
        return np.setdiff1d(array_0_rows, array_1_rows).view(array_0.dtype).reshape(-1, array_0.shape[1])

    def updateSubContainers(self):
        new_subcontainers = set()
        for sub_container in self.sub_containers:
            if not sub_container.checkSubContainer(self.box_bound):
                node_buffer = [sub_container]
                while len(node_buffer):
                    node = node_buffer[0]
                    node_buffer.remove(node)
                    for child in node.children:
                        if not child.checkSubContainer(self.box_bound):
                            node_buffer.append(child)
                        else:
                            new_subcontainers.add(child)
            else:
                new_subcontainers.add(sub_container)
        self.sub_containers = new_subcontainers

        # for itemLocation in self.root_ems.placed_items:
        #     for sub_container in self.sub_containers:
        #         self.checkInternalNodes(sub_container, itemLocation)

        for sub_container in self.sub_containers:
            all_items = np.array(self.root_ems.placed_items, dtype=float)
            # find intersections
            subc_vec = sub_container.node_vec
            intersect = np.around(np.minimum(
                        np.concatenate((-1 * np.ones(3), np.ones(3)), axis=0) * subc_vec, 
                        np.concatenate((-1 * np.ones(3), np.ones(3)), axis=0) * all_items), decimals=3)
            signal3D = (intersect[:, 0] + intersect[:, 3] > 0) * (intersect[:, 1] + intersect[:, 4] > 0) * (intersect[:, 2] + intersect[:, 5] > 0)
            idxes = np.where(signal3D.squeeze())[0]
            missing_items = self.calArray2dDiff(all_items[idxes], np.array(sub_container.placed_items, dtype=float))
            for itemLocation in missing_items:
                self.checkInternalNodes(sub_container, tuple(itemLocation))

    def updateSubContainersCheb(self, itemLocation):
        new_subcontainers = set()

        itemLocation  = np.array(itemLocation)
        max_item_size = np.max(itemLocation[3:6] - itemLocation[0:3])

        top_K = 50
        all_ems = np.array([leaf_ems.node_vec for leaf_ems in self.leaf_emss])

        # for ems in all_ems:
        #     chebyshev_distance_3d = all_ems[:,0:3] - ems[0:3]
        #     chebyshev_distance_3d = chebyshev_distance_3d[np.all(chebyshev_distance_3d > 0, axis=1)]
        #     inner_top_K = min(len(chebyshev_distance_3d), top_K)
        #     sub_container = ems.copy()
        #     if inner_top_K != 0:
        #         chebyshev_distance_3d = np.max(chebyshev_distance_3d, axis=1)
        #         selected_distance     = np.argsort(chebyshev_distance_3d)[inner_top_K-1]
        #         sub_container[3:6] = np.minimum(sub_container[0:3] + max_item_size + selected_distance, self.plain_size)
        #         sub_container = np.around(sub_container, item_decimals)
        #     sub_container = EMSNode(None, sub_container, self.low_bound)
        #     sub_container.placed_items = np.array(self.all_itemLocations)
        #     new_subcontainers.add(sub_container)

        for space in self.all_itemLocations:
            chebyshev_distance_3d = all_ems[:,0:3] - space[0:3]
            chebyshev_distance_3d = chebyshev_distance_3d[np.all(chebyshev_distance_3d > 0, axis=1)]
            inner_top_K = min(len(chebyshev_distance_3d), top_K)
            sub_container = space.copy()
            if inner_top_K == 0: continue
            chebyshev_distance_3d = np.max(chebyshev_distance_3d, axis=1)
            selected_distance     = np.argsort(chebyshev_distance_3d)[inner_top_K-1]
            sub_container[3:6] = np.minimum(sub_container[0:3] + max_item_size + selected_distance, self.plain_size)
            sub_container = np.around(sub_container, item_decimals)

            sub_container_node = EMSNode(None, sub_container, self.low_bound)
            if len(self.all_itemLocations) != 0:
                all_itemLocations = np.array(self.all_itemLocations)
                intersection_check = np.concatenate([sub_container[0:3] - all_itemLocations[:, 3:6],
                                                    all_itemLocations[:, 0:3] - sub_container[3:6]], axis=1)
                where_intersect = np.where(np.all(intersection_check < 0, axis=1))[0]
                sub_container_node.placed_items = all_itemLocations[where_intersect]
            new_subcontainers.add(sub_container_node)

        if len(new_subcontainers) != 0:
            self.sub_containers = new_subcontainers


    # Convert EMS to placement (leaf node) for the current item.
    # all real size here
    def EMSPoint(self, next_box, setting):
        posVec = set()
        if setting == 2: orientation = 6
        else: orientation = 2

        for valid_ems in self.leaf_emss:
            ems = valid_ems.node_vec
            tmp_posVec = set()
            for rot in range(orientation):  # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if abs(sizex - sizey) < tolerance:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if abs(sizex - sizey) < tolerance and abs(sizey - sizez) < tolerance:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if abs(sizex - sizey) < tolerance and abs(sizey - sizez) < tolerance:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if abs(sizex - sizey) < tolerance:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if abs(sizex - sizey) < tolerance:
                        continue

                if ems[3] - ems[0] + tolerance >= sizex and ems[4] - ems[1] + tolerance >= sizey and ems[5] - ems[
                    2] + tolerance >= sizez:
                    # TODO this part can be simplified
                    poses = []

                    # poses.append(tuple(np.around([ems[0], ems[1], ems[2], ems[0] + sizex, ems[1] + sizey, ems[2] + sizez], item_decimals)))
                    poses.append(tuple([round(x, item_decimals) for x in [ems[0], ems[1], ems[2], ems[0] + sizex, ems[1] + sizey, ems[2] + sizez]]))
                    if self.args.with_four_corner:
                        poses.append(tuple([round(x, item_decimals) for x in [ems[3] - sizex, ems[1], ems[2], ems[3], ems[1] + sizey, ems[2] + sizez]]))
                        poses.append(tuple([round(x, item_decimals) for x in [ems[0], ems[4] - sizey, ems[2], ems[0] + sizex, ems[4], ems[2] + sizez]]))
                        poses.append(tuple([round(x, item_decimals) for x in [ems[3] - sizex, ems[4] - sizey, ems[2], ems[3], ems[4], ems[2] + sizez]]))

                    for pos in poses:
                        tmp_posVec.add(pos +(0,0,1))
                        posVec.add(pos)

            valid_ems.leaf_nodes = list(tmp_posVec)

        posVec = np.array(list(posVec))
        return posVec

    def EventPoint(self, next_box, setting):
            allPostion = []
            if setting == 2: orientation = 6
            else: orientation = 2
            for k in self.ZMAP.keys():
                posVec = set()
                validEms = []

                for emsIdx in range(len(self.leaf_emss)):
                    ems = self.leaf_emss[emsIdx].node_vec
                    if abs(ems[2] - k) < tolerance:
                        validEms.append([ems[0], ems[1], -1, ems[3], ems[4], -1])

                if len(validEms) == 0:
                    continue
                validEms = np.array(validEms)
                r = self.ZMAP[k]

                for rot in range(orientation): # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
                    if rot == 0:
                        sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                    elif rot == 1:
                        sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                        if abs(sizex - sizey) < tolerance:
                            continue
                    elif rot == 2:
                        sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                        if abs(sizex - sizey) < tolerance and abs(sizey - sizez) < tolerance:
                            continue
                    elif rot == 3:
                        sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                        if abs(sizex - sizey) < tolerance and abs(sizey - sizez) < tolerance:
                            continue
                    elif rot == 4:
                        sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                        if abs(sizex - sizey) < tolerance:
                            continue
                    elif rot == 5:
                        sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                        if abs(sizex - sizey) < tolerance:
                            continue

                    for xs in r['x_up']:
                        for ys in r['y_left']:
                            xe = xs + sizex
                            ye = ys + sizey
                            posVec.add((xs, ys, k, xe, ye, k + sizez))

                        for ye in r['y_right']:
                            ys = ye - sizey
                            xe = xs + sizex
                            posVec.add((xs, ys, k, xe, ye, k + sizez))

                    for xe in r['x_bottom']:
                        xs = xe - sizex
                        for ys in r['y_left']:
                            ye = ys + sizey
                            posVec.add((xs, ys, k, xe, ye, k + sizez))

                        for ye in r['y_right']:
                            ys = ye - sizey
                            posVec.add((xs, ys, k, xe, ye, k + sizez))
                posVec = np.array(list(posVec))
                emsSize = validEms.shape[0]

                cmpPos = posVec.repeat(emsSize, axis=0)

                cmpPos = cmpPos.reshape((-1, *validEms.shape))
                cmpPos = cmpPos - validEms

                cmpPos[:, :, 3] *= -1
                cmpPos[:, :, 4] *= -1
                cmpPos = np.where(cmpPos + tolerance > 0, 1, 0)

                cmpPos = cmpPos.cumprod(axis=2)
                cmpPos = cmpPos[:, :, -1]
                cmpPos = np.sum(cmpPos, axis=1)
                validIdx = np.argwhere(cmpPos > 0)
                tmpVec = np.around(posVec[validIdx, :].squeeze(axis=1), 6)
                if len(tmpVec) != 0:
                    tmpVec = np.unique(tmpVec, axis=0)
                allPostion.extend(tmpVec.tolist())
            return allPostion  #


