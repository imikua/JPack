import numpy as np
from functools import reduce
import copy

from .convex_hull import ConvexHull, point_in_polygen


class Stack(object):
    def __init__(self, centre, mass):
        self.centre = centre
        self.mass = mass

class DownEdge(object):
    def __init__(self):
        self.box = None
        self.area = None
        self.centre2D = None
    def set_attributes(self, key, value):
        setattr(self, key, value)

class Node(object):
    def __init__(self, x, y, z, box_order):
        self.x = x
        self.y = y
        self.z = z
        self.box_order = box_order

    @property
    def attributes(self):
        return [self.x, self.y, self.z, self.box_order]

class Edge(object):
    def __init__(self, p1:Node, p2:Node, box_order):
        if p1.x <= p2.x and p1.y <= p2.y:
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = p2
            self.p2 = p1

        self.box_order = box_order



class Box(object):
    def __init__(self, width, length, height, x, y, z, density, box_order, virtual=False):
        self.w = width
        self.l = length
        self.h = height
        self.density = density

        self.x = x
        self.y = y
        self.z = z

        self.centre = np.array([self.x + self.w / 2, self.y + self.l/2, self.z + self.h/2])
        self.vertex_low = np.array([self.x, self.y, self.z])
        self.vertex_high = np.array([self.x+self.w, self.y+self.l, self.z+self.h])
        self.mass = width * length * height * density

        self.bottom_edges = []
        self.bottom_contact_area = None

        self.up_edges = {}
        self.up_virtual_edges = {}

        self.Stack = Stack(self.centre, self.mass)
        self.VirtualStack = Stack(self.centre, self.mass)
        self.involved = False

        self.box_order = box_order


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

        new_stack_centre /= (new_stack_mass+1e-6)
        if virtual:
            self.VirtualStack.mass = new_stack_mass
            self.VirtualStack.centre = new_stack_centre
        else:
            self.Stack.mass = new_stack_mass
            self.Stack.centre = new_stack_centre

    def calculated_impact(self):
        if len(self.bottom_edges) == 0 or abs(self.z) < 1e-6:
            return True
        elif not point_in_polygen(self.Stack.centre[0:2],
                                  self.bottom_contact_area):
            return False
        else:
            if len(self.bottom_edges) == 1:
                stack = self.Stack
                self.bottom_edges[0].box.up_edges[self] = stack
                self.bottom_edges[0].box.calculate_new_com()
                if not self.bottom_edges[0].box.calculated_impact():
                    return False
            else:
                direct_edge = None
                for e in self.bottom_edges:
                    if self.Stack.centre[0] > e.area[0] and self.Stack.centre[0] < e.area[2] \
                            and self.Stack.centre[1] > e.area[1] and self.Stack.centre[1] < e.area[3]:
                        direct_edge = e
                        break

                if direct_edge is not None:
                    for edge in self.bottom_edges:
                        if edge == direct_edge:
                            edge.box.up_edges[self] = self.Stack
                            edge.box.calculate_new_com()
                        else:
                            edge.box.up_edges[self] = Stack(self.Stack.centre, 0)
                            edge.box.calculate_new_com()

                    for edge in self.bottom_edges:
                        if not edge.box.calculated_impact():
                            return False

                elif len(self.bottom_edges) == 2:
                    com2D = self.Stack.centre[0:2]

                    tri_base_line = self.bottom_edges[0].centre2D - self.bottom_edges[1].centre2D
                    tri_base_len = np.linalg.norm(tri_base_line)
                    tri_base_line /= tri_base_len ** 2

                    ratio0 = abs(np.dot(com2D - self.bottom_edges[1].centre2D, tri_base_line))
                    ratio1 = abs(np.dot(com2D - self.bottom_edges[0].centre2D, tri_base_line))

                    com0 = np.array([*self.bottom_edges[0].centre2D, self.Stack.centre[2]])
                    com1 = np.array([*self.bottom_edges[1].centre2D, self.Stack.centre[2]])

                    stack0 = Stack(com0, self.Stack.mass * ratio0)
                    stack1 = Stack(com1, self.Stack.mass * ratio1)

                    self.bottom_edges[0].box.up_edges[self] = stack0
                    self.bottom_edges[0].box.calculate_new_com()

                    self.bottom_edges[1].box.up_edges[self] = stack1
                    self.bottom_edges[1].box.calculate_new_com()

                    if not self.bottom_edges[0].box.calculated_impact():
                        return False
                    if not self.bottom_edges[1].box.calculated_impact():
                        return False

                else:
                    com2D = self.Stack.centre[0:2]
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
                        newAdded_mass = self.Stack.mass * assgin_ratio[i][0]
                        newAdded_com = np.array([*e.centre2D, self.Stack.centre[2]])
                        e.box.up_edges[self] = Stack(newAdded_com, newAdded_mass)
                        e.box.calculate_new_com()

                    for e in self.bottom_edges:
                        if not e.box.calculated_impact():
                            return False
            return True

    def calculated_impact_virtual(self, first=False):
        self.involved = True
        if len(self.bottom_edges) == 0 or abs(self.z) < 1e-6:
            # if there is no box in the bottom, the position is feasible
            self.involved = False
            return True
        elif not point_in_polygen(self.VirtualStack.centre[0:2], self.bottom_contact_area):
            # if the center of mass does not fall into the polygon, the position is not feasible
            self.involved = False
            return False
        else:
            # the the center of mass falls into the polygon
            if len(self.bottom_edges) == 1:
                # there is one box in the bottom
                stack = self.VirtualStack
                self.bottom_edges[0].box.up_virtual_edges[self] = stack  # update the virtual edge of that box
                self.bottom_edges[0].box.calculate_new_com(True)         # update the center of mass of that box
                if not self.bottom_edges[0].box.calculated_impact_virtual():
                    self.involved = False
                    return False
            else:
                direct_edge = None
                for e in self.bottom_edges:
                    if e.area[2] > self.VirtualStack.centre[0] > e.area[0] and \
                            e.area[3] > self.VirtualStack.centre[1] > e.area[1]:
                        direct_edge = e
                        break

                if direct_edge is not None:
                    for edge in self.bottom_edges:
                        if edge == direct_edge:
                            edge.box.up_virtual_edges[self] = self.VirtualStack
                            edge.box.calculate_new_com(True)
                        else:
                            edge.box.up_virtual_edges[self] = Stack(self.centre, 0)
                            edge.box.calculate_new_com(True)

                    for edge in self.bottom_edges:
                        if not edge.box.calculated_impact_virtual():
                            self.involved = False
                            return False

                elif len(self.bottom_edges) == 2:
                    com2D = self.VirtualStack.centre[0:2]

                    tri_base_line = self.bottom_edges[0].centre2D - self.bottom_edges[1].centre2D
                    tri_base_len = np.linalg.norm(tri_base_line)
                    tri_base_line /= tri_base_len ** 2

                    ratio0 = abs(np.dot(com2D - self.bottom_edges[1].centre2D, tri_base_line))
                    ratio1 = abs(np.dot(com2D - self.bottom_edges[0].centre2D, tri_base_line))

                    com0 = np.array([*self.bottom_edges[0].centre2D, self.VirtualStack.centre[2]])
                    com1 = np.array([*self.bottom_edges[1].centre2D, self.VirtualStack.centre[2]])

                    stack0 = Stack(com0, self.VirtualStack.mass * ratio0)
                    stack1 = Stack(com1, self.VirtualStack.mass * ratio1)

                    self.bottom_edges[0].box.up_virtual_edges[self] = stack0
                    self.bottom_edges[0].box.calculate_new_com(True)
                    self.bottom_edges[1].box.up_virtual_edges[self] = stack1
                    self.bottom_edges[1].box.calculate_new_com(True)

                    if not self.bottom_edges[0].box.calculated_impact_virtual() \
                            or not self.bottom_edges[1].box.calculated_impact_virtual():
                        self.involved = False
                        return False

                else:
                    com2D = self.VirtualStack.centre[0:2]
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
                        newAdded_mass = self.VirtualStack.mass * assgin_ratio[i][0]
                        newAdded_com = np.array([*e.centre2D, self.VirtualStack.centre[2]])
                        e.box.up_virtual_edges[self] = Stack(newAdded_com, newAdded_mass)
                        e.box.calculate_new_com(True)

                    for e in self.bottom_edges:
                        if not e.box.calculated_impact_virtual():
                            self.involved = False
                            return False

            if first:
                # only use one time
                for e in self.bottom_edges:
                    e.box.up_virtual_edges.pop(self)
            self.involved = False
            return True




class Space(object):
    def __init__(self, width=10, length=10, height=10, minimum_box_size=0, maximum_box_size=0, num_box=40, node_dim=9):

        self.plain_size = np.array([width, length, height])
        self.max_axis = max(width, length)
        self.height = height
        self.minimum_box_size = minimum_box_size
        self.maximum_box_size = maximum_box_size
        self.low_bound = minimum_box_size
        self.node_dim = node_dim
        self.num_box = num_box

        self.base_box = Box(width, length, 0, 0, 0, 0, 0, 0)
        self.box_vec = np.zeros((1, node_dim))
        self.box_vec[0] = [0, 0, 0, width, length, 0, 0, 0., 1.]
        self.boxes = [self.base_box]
        self.box_idx = 1

        # upLetter stores 2D projection bounds for already placed boxes.
        # Need capacity for base_box (idx=0) + up to num_box placed boxes (idx=1..num_box).
        self.upLetter = np.zeros((num_box + 1, 5))
        self.EMS = np.zeros((1000, 6))
        self.NOEMS = 1

        self.IP = [
            [0, 0],
            [0, self.plain_size[1]],
            [self.plain_size[0], 0],
            [self.plain_size[0], self.plain_size[1]]
        ]

        self.reset()

    def reset(self):
        self.base_box = Box(self.plain_size[0], self.plain_size[1], 0, 0, 0, 0, 0, 0)
        self.box_vec = np.zeros((1, self.node_dim))
        self.box_vec[0] = [0, 0, 0, self.plain_size[0], self.plain_size[1], 0, 0, 0., 1.]
        self.upLetter = np.zeros((self.num_box + 1, 5))
        self.upLetter[0] = [0, 0, self.plain_size[0], self.plain_size[1], 0]

        self.boxes = [self.base_box]
        self.box_idx = 1
        self.EMS = np.zeros((1000, 6))
        self.EMS[0] = np.array([0, 0, 0, *self.plain_size])
        self.NOEMS = 1

        self.IP = [
            [0, 0],
            [0, self.plain_size[1]],
            [self.plain_size[0], 0],
            [self.plain_size[0], self.plain_size[1]]
        ]

    def get_ratio(self):
        box_volumn = reduce(lambda x,y: x+y, [box.w * box.l * box.h for box in self.boxes], 0.0)
        bin_volumn = np.prod(self.plain_size)
        ratio = box_volumn / bin_volumn
        #assert ratio <= 1.0
        return ratio

    def scale_down(self, bottom_contact_area):
        centre2D = np.mean(bottom_contact_area, axis=0)
        direction2D = bottom_contact_area - centre2D
        bottom_contact_area -= direction2D * 0.1
        return bottom_contact_area.tolist()

    def interSect2D(self, box):
        if self.box_idx == 1:
            return 0, [], []
        intersect = np.around(np.minimum(box, self.upLetter[1:self.box_idx]), 6)
        signal = (intersect[:, 0] + intersect[:, 2] > 0) * (intersect[:, 1] + intersect[:, 3] > 0)
        index = np.where(signal)[0]
        if len(index) == 0:
            return 0, [], []
        else:
            return np.max(self.upLetter[index, 4]), index, intersect[index]


    def drop_box(self, box, pos, density, setting):
        x, y = pos
        w, l, h = box

        if x + w - 1e-6 > self.plain_size[0] or y + l - 1e-6 > self.plain_size[1]:
            return False
        if x + 1e-6 < 0 or y + 1e-6 < 0:
            return False

        box_info = np.array([-x, -y, x+w, y+l, 0])
        max_h, interIdx, interArea = self.interSect2D(box_info)

        if max_h + h - 1e-6 > self.height:
            return False
        box_info = np.array([-x, -y, x+w, y+l, max_h+h])

        box_now = Box(w, l, h, x, y, max_h, density, self.box_idx)

        if setting == 1 or setting == 3:
            combine_contact_points = []
            for inner in range(len(interIdx)):
                idx = interIdx[inner]
                tmp = self.boxes[idx]
                if abs(tmp.z + tmp.h - max_h) < 1e-6:
                    x1, y1, x2, y2, _ = interArea[inner]
                    x1, y1 = -x1, -y1

                    newEdge = DownEdge()
                    newEdge.set_attributes('box', tmp)
                    newEdge.set_attributes('area', (x1, y1, x2, y2))
                    newEdge.set_attributes('centre2D', np.array([x1 + x2, y1 + y2]) / 2)
                    box_now.bottom_edges.append(newEdge)
                    combine_contact_points.append([x1, y1])
                    combine_contact_points.append([x1, y2])
                    combine_contact_points.append([x2, y1])
                    combine_contact_points.append([x2, y2])

            if len(combine_contact_points) > 0:
                box_now.bottom_contact_area = self.scale_down(ConvexHull(combine_contact_points))

        isFeasible = self.check_box(w, l, h, x, y, max_h, box_now, setting)
        if isFeasible:
            self.boxes.append(box_now)
            self.upLetter[self.box_idx] = box_info
            self.box_vec = np.concatenate(
                (self.box_vec, [[x, y, max_h, w, l, h, density, 0., 1.]]), axis=0,
            )
            self.box_idx += 1
            return True
        return False


    def drop_box_virtual(self, box_size, position, density, box_order, setting,):
        w, l, h = box_size
        x, y = position

        checkResult = True
        if x + w - 1e-6 > self.plain_size[0] or y + l - 1e-6 > self.plain_size[1]:
            checkResult = False
        if x + 1e-6 < 0 or y + 1e-6 < 0:
            checkResult = False

        box_info = np.array([-x, -y, x+w, y+l, 0])
        max_h, interIdx, interArea = self.interSect2D(box_info)

        if max_h + h - 1e-6 > self.height:
            checkResult = False

        box_now = Box(w, l, h, x, y, max_h, density, box_order, True)

        if (setting == 1 or setting == 3) and checkResult:
            combine_contact_points = []
            for inner in range(len(interIdx)):
                idx = interIdx[inner]
                tmp = self.boxes[idx]
                if abs(tmp.z + tmp.h - max_h) < 1e-6:
                    x1, y1, x2, y2, _ = interArea[inner]
                    x1, y1 = -x1, -y1

                    newEdge = DownEdge()
                    newEdge.set_attributes('box', tmp)
                    newEdge.set_attributes('area', (x1, y1, x2, y2))
                    newEdge.set_attributes('centre2D', np.array([x1 + x2, y1 + y2]) / 2)
                    box_now.bottom_edges.append(newEdge)
                    combine_contact_points.append([x1, y1])
                    combine_contact_points.append([x1, y2])
                    combine_contact_points.append([x2, y1])
                    combine_contact_points.append([x2, y2])

            if len(combine_contact_points) > 0:
                box_now.bottom_contact_area = self.scale_down(ConvexHull(combine_contact_points))

        return (checkResult and self.check_box(w, l, h, x, y, max_h, box_now, setting, True)), max_h

    def check_box(self, w, l, h, x, y, z, box_now, setting, virtual=False):
        assert isinstance(setting, int), 'The environment setting should be integer.'

        if x + w - 1e-6 > self.plain_size[0] or y + l - 1e-6 > self.plain_size[1]:
            return False
        if x + 1e-6 < 0 or y + 1e-6 < 0:
            return False
        if z + h - 1e-6 > self.height:
            return False

        if setting == 2:
            return True
        else:
            if abs(z) < 1e-6:
                return True
            if not virtual:
                return box_now.calculated_impact()
            else:
                return box_now.calculated_impact_virtual(True)

    def GENIP(self, itemArea):
        x1, y1, x2, y2 = itemArea

        dupIP = np.concatenate((self.IP, self.IP), axis=1)
        inner = dupIP - itemArea

        former = (inner[:, 0] > 0) * (inner[:, 1] > 0)
        latter = (inner[:, 2] < 0) * (inner[:, 3] < 0)
        mask = np.where(1 - (former == 1) * (latter == 1))[0]
        self.IP = np.array(self.IP)[mask]

        self.update_duplicate_points([x1, y1])
        self.update_duplicate_points([x1, y2])
        self.update_duplicate_points([x2, y1])
        self.update_duplicate_points([x2, y2])

        return

    def update_duplicate_points(self, point):
        dup = self.IP - point
        mask = np.where((abs(dup[:, 0]) <= 1e-6) * (abs(dup[:, 1]) <= 1e-6))[0]
        if len(mask) == 0:
            self.IP = np.concatenate((self.IP, [point]), axis=0)

        return


    def IntersecPoint(self, next_box, setting):
        posVec = set()
        if setting == 2:
            orientation = 6
        else:
            orientation = 2

        for ip in self.IP:
            for rot in range(orientation):
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if abs(sizex - sizey) < 1e-6:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if abs(sizex - sizey) < 1e-6 and abs(sizey - sizez) < 1e-6:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if abs(sizex - sizey) < 1e-6 and abs(sizey - sizez) < 1e-6:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if abs(sizex - sizey) < 1e-6:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if abs(sizex - sizey) < 1e-6:
                        continue

                if ip[0] + sizex <= self.plain_size[0] and ip[1] + sizey <= self.plain_size[1]:
                    posVec.add((ip[0], ip[1], 0, ip[0] + sizex, ip[1] + sizey, sizez))
                if ip[0] - sizex >= 0 and ip[1] + sizey <= self.plain_size[1]:
                    posVec.add((ip[0] - sizex, ip[1], 0, ip[0], ip[1] + sizey, sizez))
                if ip[0] + sizex <= self.plain_size[0] and ip[1] - sizey >= 0:
                    posVec.add((ip[0], ip[1] - sizey, 0, ip[0] + sizex, ip[1], sizez))
                if ip[0] - sizex >= 0 and ip[1] - sizey >= 0:
                    posVec.add((ip[0] - sizex, ip[1] - sizey, 0, ip[0], ip[1], sizez))

        posVec = np.array(list(posVec))
        return posVec


    def EMSPoint(self, next_box, setting):
        posVec = set()
        if setting == 2:
            orientation = 6
        else:
            orientation = 2

        for ems in self.EMS:
            for rot in range(orientation):
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if abs(sizex - sizey) < 1e-6:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if abs(sizex - sizey) < 1e-6 and abs(sizey - sizez) < 1e-6:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if abs(sizex - sizey) < 1e-6 and abs(sizey - sizez) < 1e-6:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if abs(sizex - sizey) < 1e-6:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if abs(sizex - sizey) < 1e-6:
                        continue

                if ems[3] - ems[0] + 1e-6 >= sizex and ems[4] - ems[1] + 1e-6 >= sizey and ems[5] - ems[
                    2] + 1e-6 >= sizez:
                    posVec.add((ems[0], ems[1], ems[2], ems[0] + sizex, ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[1], ems[2], ems[3], ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[0], ems[4] - sizey, ems[2], ems[0] + sizex, ems[4], ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[4] - sizey, ems[2], ems[3], ems[4], ems[2] + sizez))
        posVec = np.array(list(posVec))
        return posVec

    def GENEMS(self, itemLocation):
        originemss = self.NOEMS
        delflag, validflag, intersect = self.interSectEMS3D(np.array(itemLocation))

        for idx in range(len(delflag)):
            emsIdx = delflag[idx]
            inter = intersect[idx]
            self.Difference(emsIdx, inter)

        if len(delflag) != 0:
            validflag = [*validflag, *range(originemss, self.NOEMS)]
            validLength = len(validflag)
            self.EMS[0:validLength, :] = self.EMS[validflag, :]
            self.EMS[validLength:self.NOEMS, :] = 0
            self.NOEMS = validLength

        self.NOEMS, self.EMS = self.EliminateInscribedEMS(self.NOEMS, self.EMS)

    def interSectEMS3D(self, itemLocation):
        itemLocation[0:3] *= -1

        EMS = self.EMS[0:self.NOEMS].copy()
        EMS[:, 0:3] *= -1

        if self.box_idx == 1:
            return 0, [], []

        intersect = np.around(np.minimum(itemLocation, EMS), 6)
        signal = (intersect[:, 0] + intersect[:, 3] > 0) * (intersect[:, 1] + intersect[:, 4] > 0) * (intersect[:, 2] + intersect[:, 5] > 0)
        delindex = np.where(signal)[0]
        saveindex = np.where(signal == False)[0]
        intersect = intersect[delindex]
        intersect[:, 0:3] *= -1
        return delindex, saveindex, intersect

    def Difference(self, emsID, intersection):
        x1, y1, z1, x2, y2, z2 = self.EMS[emsID]
        x3, y3, z3, x4, y4, z4, = intersection
        if self.IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z1, x3, y2, z2):
            self.AddNewEMS(x1, y1, z1, x3, y2, z2)
        if self.IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x4, y1, z1, x2, y2, z2):
            self.AddNewEMS(x4, y1, z1, x2, y2, z2)
        if self.IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z1, x2, y3, z2):
            self.AddNewEMS(x1, y1, z1, x2, y3, z2)
        if self.IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y4, z1, x2, y2, z2):
            self.AddNewEMS(x1, y4, z1, x2, y2, z2)
        if self.IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z4, x2, y2, z2):
            self.AddNewEMS(x1, y1, z4, x2, y2, z2)

    def IsUsableEMS(self, xlow, ylow, zlow, x1, y1, z1, x2, y2, z2):
        if ((x2 - x1 + 1e-6 >= xlow) and (y2 - y1 + 1e-6 >= ylow) and (z2 - z1 + 1e-6 >= zlow)):
            return True
        return False

    def AddNewEMS(self, a, b, c, x, y, z):
        self.EMS[self.NOEMS] = np.array([a, b, c, x, y, z])
        self.NOEMS += 1

    def EliminateInscribedEMS(self, NOEMS, EMS):
        delflags = np.zeros(NOEMS)
        for i in range(NOEMS):
            for j in range(NOEMS):
                if i == j:
                    continue
                if (EMS[i][0] >= EMS[j][0] and EMS[i][1] >= EMS[j][1]
                    and EMS[i][2] >= EMS[j][2] and EMS[i][3] <= EMS[j][3]
                    and EMS[i][4] <= EMS[j][4] and EMS[i][5] <= EMS[j][5]):
                    delflags[i] = 1
                    break
        saveIdx = np.where(delflags == 0)[0]
        validLength = len(saveIdx)

        if validLength!= 0:
            EMS[0:validLength] = EMS[saveIdx]
        EMS[validLength:NOEMS] = 0
        NOEMS = validLength
        return NOEMS, EMS





