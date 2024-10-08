import numpy as np


class Hyperbox:
    def __init__(self, v, w, classifier, theta):
        self.Min = v
        self.Max = w
        self.Center = (v + w) / 2
        self.classifier = classifier
        self.theta = theta

    def is_expandable(self, x, theta=-1):
        if theta == -1:
            theta = self.theta
        candidV = np.minimum(self.Min, x)
        candidW = np.maximum(self.Max, x)
        return all((candidW - candidV) < theta)

    def expand(self, point):
        self.Min = np.minimum(self.Min, point)
        self.Max = np.maximum(self.Max, point)
        self.Center = (self.Min + self.Max) / 2

    def overlaps(self, other, overlap_threshold):
        overlap_min = np.maximum(self.Min, other.Min)
        overlap_max = np.minimum(self.Max, other.Max)
        overlap_volume = np.prod(np.maximum(0, overlap_max - overlap_min))
        self_volume = np.prod(self.Max - self.Min)
        other_volume = np.prod(other.Max - other.Min)
        if self_volume == 0 or other_volume == 0:
            return False
        return (overlap_volume / self_volume >= overlap_threshold) and (
                    overlap_volume / other_volume >= overlap_threshold)

    def is_overlapped(self, other):
        minW = np.minimum(self.Max, other.Max)
        maxV = np.maximum(self.Min, other.Min)
        return all(maxV < minW)

    def contract(self, conBoxes):
        ndimension = len(self.Min)
        for box in conBoxes:
            if self.type == box.type:
                print("Type of boxes are the same")
                continue
            if not self.is_overlapped(box):
                continue
            minOverlap = np.inf
            dimOverlap = -1
            for n in range(ndimension):
                if (self.Min[n] <= box.Min[n] and box.Min[n] < self.Max[n] and self.Max[n] <= box.Max[n]):
                    if (self.Max[n] - box.Min[n]) < minOverlap:
                        minOverlap = self.Max[n] - box.Min[n]
                        type = 1
                        dimOverlap = n

                elif box.Min[n] <= self.Min[n] and self.Min[n] < box.Max[n] and box.Max[n] <= self.Max[n]:
                    if (box.Max[n] - self.Min[n]) < minOverlap:
                        minOverlap = box.Max[n] - self.Min[n]
                        type = 2
                        dimOverlap = n

                elif self.Min[n] <= box.Min[n] and box.Max[n] <= self.Max[n]:
                    m = min((box.Min[n] - self.Min[n]), (self.Max[n] - box.Max[n]))
                    if m < minOverlap:
                        minOverlap = m
                        type = 3
                        dimOverlap = n

                elif box.Min[n] <= self.Min[n] and self.Max[n] <= box.Max[n]:
                    m = min((self.Min[n] - box.Min[n]), (box.Max[n] - self.Max[n]))
                    if m < minOverlap:
                        minOverlap = m
                        type = 4
                        dimOverlap = n

            if type == 1:
                box.Min[dimOverlap] = (self.Max[dimOverlap] + box.Min[dimOverlap]) / 2
                self.Max[dimOverlap] = box.Min[dimOverlap]

            elif type == 2:
                self.Min[dimOverlap] = (box.Max[dimOverlap] + self.Min[dimOverlap]) / 2
                box.Max[dimOverlap] = self.Min[dimOverlap]

            elif type == 3:
                if (box.Max[dimOverlap] - self.Min[dimOverlap]) < (self.Max[dimOverlap] - box.Min[dimOverlap]):
                    self.Min[dimOverlap] = box.Max[dimOverlap]
                else:
                    self.Max[dimOverlap] = box.Min[dimOverlap]

            else:
                if (self.Max[dimOverlap] - box.Min[dimOverlap]) < (box.Max[dimOverlap] - self.Min[dimOverlap]):
                    box.Min[dimOverlap] = self.Max[dimOverlap]
                else:
                    box.Max[dimOverlap] = self.Min[dimOverlap]

    def membership(self, x):
        disvec = np.abs(self.Center - x)
        halfsize = (self.Max - self.Min) / 2
        d = disvec - halfsize
        m = np.linalg.norm(d[d > 0])
        m = m / np.sqrt(len(x))  # adapting with high dimensional problems
        m = 1 - m
        m = np.power(m, 4)
        return m

    def f(self, r, y):
        if r * y > 1:
            return 1
        if r * y >= 0:
            return r * y
        return 0
