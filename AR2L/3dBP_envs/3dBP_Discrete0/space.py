import numpy as np
from functools import reduce
import copy

try:
    from .convex_hull import ConvexHull, point_in_polygen
except:
    from convex_hull import ConvexHull, point_in_polygen


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
        if len(self.bottom_edges) == 0 or self.z == 0:
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
        if len(self.bottom_edges) == 0 or self.z == 0:
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

        self.base_box = Box(width, length, 0, 0, 0, 0, 0, 0)
        self.plain = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        self.box_vec = np.zeros((1, node_dim))
        self.box_vec[0] = [0, 0, 0, width, length, 0, 0, 0., 1.]
        self.boxes = [self.base_box]
        self.box_idx = 1
        self.EMS = [np.array([0, 0, 0, *self.plain_size])]

        self.reset()

    def reset(self):
        self.base_box = Box(self.plain_size[0], self.plain_size[1], 0, 0, 0, 0, 0, 0)
        self.plain[...] = 0
        self.box_vec = np.zeros((1, self.node_dim))
        self.box_vec[0] = [0, 0, 0, self.plain_size[0], self.plain_size[1], 0, 0, 0., 1.]

        self.boxes = [self.base_box]
        self.box_idx = 1
        self.EMS = [np.array([0, 0, 0, *self.plain_size])]

    def get_ratio(self):
        box_volumn = reduce(lambda x,y: x+y, [box.w * box.l * box.h for box in self.boxes], 0.0)
        bin_volumn = np.prod(self.plain_size)
        ratio = box_volumn / bin_volumn
        assert ratio <= 1.0
        return ratio

    def scale_down(self, bottom_contact_area):
        centre2D = np.mean(bottom_contact_area, axis=0)
        direction2D = bottom_contact_area - centre2D
        bottom_contact_area -= direction2D * 0.1
        return bottom_contact_area.tolist()

    def update_height_graph(self, plain, box:Box):
        plain = copy.deepcopy(plain)
        lu = int(box.x)
        lb = int(box.x + box.w)
        ru = int(box.y)
        rb = int(box.y + box.l)
        max_h = np.max(plain[lu:lb, ru:rb])
        max_h = max(max_h, box.z + box.h)
        plain[lu:lb, ru:rb] = max_h
        return plain

    def drop_box(self, box, pos, density, setting):
        x, y = pos
        w, l, h = box

        rec = self.plain[int(x):int(x+w), int(y):int(y+l)]
        max_h = np.max(rec)
        box_now = Box(w, l, h, x, y, max_h, density, self.box_idx)

        '''
        if setting == 1 or setting == 3:
            combine_contact_points = []
            for tmp in self.boxes:
                if tmp.z + tmp.h == max_h:
                    x1 = max(box_now.vertex_low[0], tmp.vertex_low[0])
                    y1 = max(box_now.vertex_low[1], tmp.vertex_low[1])
                    x2 = min(box_now.vertex_high[0], tmp.vertex_high[0])
                    y2 = min(box_now.vertex_high[1], tmp.vertex_high[1])
                    if x1 >= x2 or y1 >= y2:
                        continue
                    else:
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
        '''

        isFeasible = self.check_box(w, l, h, x, y, max_h, box_now, setting)
        if isFeasible:
            self.boxes.append(box_now)
            self.plain = self.update_height_graph(self.plain, self.boxes[-1])
            self.height = max(self.height, max_h + h)
            self.box_vec = np.concatenate(
                (self.box_vec, [[x, y, max_h, w, l, h, density, 0., 1.]]), axis=0,
            )
            self.box_idx += 1
            return True
        return False


    def drop_box_virtual(self, box_size, position, density, box_order, setting, UpdateMap=False):
        w, l, h = box_size
        x, y = position

        rec = self.plain[int(x):int(x+w), int(y):int(y+l)]
        max_h = np.max(rec)

        box_now = Box(w, l, h, x, y, max_h, density, box_order, True)

        '''
        if setting == 1 or setting == 3:
            combine_contact_points = []
            for tmp in self.boxes:
                if tmp.z + tmp.h == max_h:
                    x1 = max(box_now.vertex_low[0], tmp.vertex_low[0])
                    y1 = max(box_now.vertex_low[1], tmp.vertex_low[1])
                    x2 = min(box_now.vertex_high[0], tmp.vertex_high[0])
                    y2 = min(box_now.vertex_high[1], tmp.vertex_high[1])
                    if x1 >= x2 or y1 >= y2:
                        continue
                    else:
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
        '''

        if UpdateMap:
            height_map = self.update_height_graph(self.plain, box_now)

        return self.check_box(w, l, h, x, y, max_h, box_now, setting, True), max_h

    def check_box(self, w, l, h, x, y, z, box_now, setting, virtual=False):
        assert isinstance(setting, int), 'The environment setting should be integer.'

        if x + w > self.plain_size[0] or y + l > self.plain_size[1]:
            return False
        if x < 0 or y < 0:
            return False
        if z + h > self.height:
            return False

        if setting == 2:
            return True
        else:
            if z == 0:
                return True
            if not virtual:
                return box_now.calculated_impact()
            else:
                return box_now.calculated_impact_virtual(True)

    def EMSPoint(self, next_box, setting):
        posVec = set()
        orientation = 6 if setting == 2 else 2

        # 0: x y z, 1: y x z, 2: x z y,  3: y z x, 4:z x y, 5: z y x
        for ems in self.EMS:
            for rot in range(orientation):
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if sizex == sizey:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if sizex == sizey:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if sizex == sizey:
                        continue

                if ems[3] - ems[0] >= sizex and ems[4] - ems[1] >= sizey and ems[5] - ems[2] >= sizez:
                    posVec.add((ems[0], ems[1], ems[2], ems[0] + sizex, ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[1], ems[2], ems[3], ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[0], ems[4] - sizey, ems[2], ems[0] + sizex, ems[4], ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[4] - sizey, ems[2], ems[3], ems[4], ems[2] + sizez))
        posVec = np.array(list(posVec))
        return posVec

    def GENEMS(self, itemLocation):
        numofemss = len(self.EMS)
        delflag = []
        for emsIdx in range(numofemss):
            xems1, yems1, zems1, xems2, yems2, zems2 = self.EMS[emsIdx]
            xtmp1, ytmp1, ztmp1, xtmp2, ytmp2, ztmp2 = itemLocation

            if (xems1 > xtmp1): xtmp1 = xems1
            if (yems1 > ytmp1): ytmp1 = yems1
            if (zems1 > ztmp1): ztmp1 = zems1
            if (xems2 < xtmp2): xtmp2 = xems2
            if (yems2 < ytmp2): ytmp2 = yems2
            if (zems2 < ztmp2): ztmp2 = zems2

            if (xtmp1 > xtmp2): xtmp1 = xtmp2
            if (ytmp1 > ytmp2): ytmp1 = ytmp2
            if (ztmp1 > ztmp2): ztmp1 = ztmp2
            if (xtmp1 == xtmp2 or ytmp1 == ytmp2 or ztmp1 == ztmp2):
                continue

            self.Difference(emsIdx, (xtmp1, ytmp1, ztmp1, xtmp2, ytmp2, ztmp2))
            delflag.append(emsIdx)

        if len(delflag) != 0:
            NOEMS = len(self.EMS)
            self.EMS = [self.EMS[i] for i in range(NOEMS) if i not in delflag]
        self.EliminateInscribedEMS()

    def Difference(self, emsID, intersection):
        x1, y1, z1, x2, y2, z2 = self.EMS[emsID]
        x3, y3, z3, x4, y4, z4, = intersection
        if self.low_bound == 0:
            self.low_bound = 0.1
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

    def AddNewEMS(self, a, b, c, x, y, z):
        self.EMS.append(np.array([a, b, c, x, y, z]))

    def EliminateInscribedEMS(self):
        NOEMS = len(self.EMS)
        delflags = np.zeros(NOEMS)
        for i in range(NOEMS):
            for j in range(NOEMS):
                if i == j:
                    continue
                if (self.EMS[i][0] >= self.EMS[j][0] and self.EMS[i][1] >= self.EMS[j][1]
                        and self.EMS[i][2] >= self.EMS[j][2] and self.EMS[i][3] <= self.EMS[j][3]
                        and self.EMS[i][4] <= self.EMS[j][4] and self.EMS[i][5] <= self.EMS[j][5]):
                    delflags[i] = 1
                    break
        self.EMS = [self.EMS[i] for i in range(NOEMS) if delflags[i] != 1]
        return len(self.EMS)

    def IsUsableEMS(self, xlow, ylow, zlow, x1, y1, z1, x2, y2, z2):
        xd = x2 - x1
        yd = y2 - y1
        zd = z2 - z1
        if ((xd >= xlow) and (yd >= ylow) and (zd >= zlow)):
            return True
        return False







