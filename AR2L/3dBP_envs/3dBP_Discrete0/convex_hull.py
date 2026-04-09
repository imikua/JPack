import numpy as np

class Point2D(object):
    def __init__(self, x, y):
        self.x, self.y = x, y


class Line2D(object):
    def __init__(self, point1:Point2D, point2:Point2D):
        self.p1 = point1
        self.p2 = point2

        if self.p1.x != self.p2.x:
            self.slope = (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
        else:
            self.slope = (self.p2.y - self.p1.y) * np.inf

    @staticmethod
    def rotation_direction(line1, line2):
        slope1 = line1.slope
        slope2 = line2.slope

        if abs(slope1) == abs(slope2) == np.inf:
            return 0

        slope_diff = slope2 - slope1
        if slope_diff > 0:
            return -1 # counter-clockwise
        elif slope_diff == 0:
            return 0  # colinear
        else:
            return 1  # clockwise


def ConvexHull(point_list):

    point_list = np.array(point_list).astype(np.float32)
    point_list[:,0] += point_list[:,1] * 1e-6
    point_list = point_list.tolist()
    upperHull = list()
    lowerHull = list()

    sorted_list = sorted(point_list, key = lambda x: x[0])

    for point in sorted_list:
        if len(lowerHull) >= 2:
            line1 = Line2D(Point2D(lowerHull[-2][0], lowerHull[-2][1]),
                           Point2D(lowerHull[-1][0], lowerHull[-1][1]))
            line2 = Line2D(Point2D(lowerHull[-1][0], lowerHull[-1][1]),
                           Point2D(point[0], point[1]))

        while len(lowerHull) >= 2 and Line2D.rotation_direction(line1, line2) != -1:
            removed = lowerHull.pop()
            if lowerHull[0] == lowerHull[-1]:
                break
            line1 = Line2D(Point2D(lowerHull[-2][0], lowerHull[-2][1]),
                           Point2D(lowerHull[-1][0], lowerHull[-1][1]))
            line2 = Line2D(Point2D(lowerHull[-1][0], lowerHull[-1][1]),
                           Point2D(point[0], point[1]))
        lowerHull.append(point)

    reverse_list = sorted_list[::-1]
    for point in reverse_list:
        if len(upperHull) >= 2:
            line1 = Line2D(Point2D(upperHull[-2][0], upperHull[-2][1]),
                           Point2D(upperHull[-1][0], upperHull[-1][1]))
            line2 = Line2D(Point2D(upperHull[-1][0], upperHull[-1][1]),
                           Point2D(point[0], point[1]))

        while len(upperHull) >= 2 and Line2D.rotation_direction(line1, line2) != -1:
            removed = upperHull.pop()
            if upperHull[0] == upperHull[-1]:
                break
            line1 = Line2D(Point2D(upperHull[-2][0], upperHull[-2][1]),
                           Point2D(upperHull[-1][0], upperHull[-1][1]))
            line2 = Line2D(Point2D(upperHull[-1][0], upperHull[-1][1]),
                           Point2D(point[0], point[1]))
        upperHull.append(point)

    removed = upperHull.pop()
    removed = lowerHull.pop()
    convexHullPoints = lowerHull + upperHull
    convexHullPoints = np.array(convexHullPoints)

    return convexHullPoints

def point_in_polygen(point, coords):
    lat, lon = point
    polysides = len(coords)
    j = polysides - 1
    oddnodes = False

    for i in range(polysides):
        if np.sum(np.cross(coords[i] - point, point - coords[j])) == 0:
            return False

        if (coords[i][1] < lon and coords[j][1] >= lon) or (coords[j][1] < lon and coords[i][1] >= lon):
            if (coords[i][0] + (lon - coords[i][1]) / (coords[j][1] - coords[i][1]) * (coords[j][0] - coords[i][0])) < lat:
                oddnodes = not oddnodes
        j = i

    return oddnodes

