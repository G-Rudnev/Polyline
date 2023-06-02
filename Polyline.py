import math
from math import inf as inf
from math import pi as pi
from math import pow as pow
from math import tan as tan
from math import sin as sin
from math import cos as cos
from math import atan as atan
from math import atan2 as atan2

import numpy as np
from numpy import matmul as matmul
from numpy.linalg import norm as norm
from numpy.linalg import inv as inv

class Polyline(np.ndarray):

    """
        Arbitrary gapped and closed polyline.\n
        The first and the last pnts should be non-zero both.\n
        Two gaps should be separated by at least one non-gap pnt
    """

    def __new__(cls, N, edgeShift, *args, **kwargs):
        obj = np.zeros([3, N])
        obj[2, :] = 1.0
        return obj.view(cls)

    def __init__(self, N : int, edgeShift : int = 0):
        """edgeShift is as it supposed by vectorization properties"""
        self.Nlines = N
        self.edgeShift = edgeShift
        self.gapsIdxs = np.zeros(self.Nlines, dtype = np.uint32)
        self.Ngaps = 0

        self.checkList = np.zeros([3, N // 2])

    def Fill(self, polyline : np.ndarray, gapsIdxs : np.ndarray, Nlines : int, Ngaps : int):
        self[:2, :Nlines] = polyline[:2, :Nlines]
        self.gapsIdxs[:Ngaps] = gapsIdxs[:Ngaps]
        self.Nlines = Nlines
        self.Ngaps = Ngaps
        return Nlines

    def GetPnt(self, i : int, increment : int = 1, homogen : bool = False):
        """
            Returns (shift, [x, y])
        """
        shift = 0
        while True:
            
            pnt = self[:(2 + homogen), (i + shift) % self.Nlines]
            if (abs(pnt[0]) > 1e-5 or abs(pnt[1]) > 1e-5):
                return (pnt, shift)

            shift += increment

    def Closest_pnt(self, pnt : np.ndarray, seg_from : int, seg_to : int, onGaps : bool = False, exceptGaps : set = set()):
        """
            Returns (seg_i, closestPnt, minDist)
        """
        minDist = inf
        seg_i = -1
        closestPnt = np.zeros([2])
        closestPnt_ = np.zeros([2])

        seg_r = np.zeros([2])
        v = np.zeros([2])
        r = np.zeros([2])

        while seg_from < seg_to:
            p1, shift = self.GetPnt(seg_from + 1)
            if (not onGaps):
                if (shift != 0):
                    seg_from += (shift + 1)
                    continue
            elif (shift == 0 or (seg_from + 1) in exceptGaps):
                seg_from += 1
                continue
            p0, _ = self.GetPnt(seg_from)

            seg_r[0] = p1[0] - p0[0]
            seg_r[1] = p1[1] - p0[1]

            v[0] = pnt[0] - p0[0]
            v[1] = pnt[1] - p0[1]

            val = np.dot(seg_r, seg_r)
            if (val < 1e-4):
                t = 0.0
            else:
                t = np.dot(v, seg_r) / val
                if (t < 0.0):
                    t = 0.0
                elif (t > 1.0):
                    t = 1.0

            closestPnt_[0] = p0[0] + t * seg_r[0]
            closestPnt_[1] = p0[1] + t * seg_r[1]
            r[0] = pnt[0] - closestPnt_[0]
            r[1] = pnt[1] - closestPnt_[1]
            
            dist = norm(r)
            if (dist < minDist - 1e-9):
                seg_i = seg_from + onGaps
                closestPnt = closestPnt_.copy()
                minDist = dist

            seg_from += (shift + 1)

        return (seg_i, closestPnt, minDist)

    def Closest_pnt00(self, seg_from : int, seg_to : int, onGaps : bool = False, exceptGaps : set = set()):
        """
            Find closest for [0.0, 0.0]
            Returns (seg_i, closestPnt, minDist)
        """
        minDist = inf
        seg_i = -1
        closestPnt = np.zeros([2])
        closestPnt_ = np.zeros([2])

        seg_r = np.zeros([2])
        v = np.zeros([2])
        r = np.zeros([2])

        while seg_from < seg_to:
            p1, shift = self.GetPnt(seg_from + 1)
            if (not onGaps):
                if (shift != 0):
                    seg_from += (shift + 1)
                    continue
            elif (shift == 0 or (seg_from + 1) in exceptGaps):
                seg_from += 1
                continue
            p0, _ = self.GetPnt(seg_from)

            seg_r[0] = p1[0] - p0[0]
            seg_r[1] = p1[1] - p0[1]

            val = np.dot(seg_r, seg_r)
            if (val < 1e-4):
                t = 0.0
            else:
                t = -np.dot(p0[:2], seg_r) / val
                if (t < 0.0):
                    t = 0.0
                elif (t > 1.0):
                    t = 1.0

            closestPnt_[0] = p0[0] + t * seg_r[0]
            closestPnt_[1] = p0[1] + t * seg_r[1]
            
            dist = norm(closestPnt_)
            if (dist < minDist - 1e-9):
                seg_i = seg_from + onGaps
                closestPnt = closestPnt_.copy()
                minDist = dist

            seg_from += (shift + 1)

        return (seg_i, closestPnt, minDist)

    def Check_segment_intersections(self, p0 : np.ndarray, p1 : np.ndarray, seg_from : int, seg_to : int, checkAll : bool = False, num : int = 0, ignoreGaps : bool = False):
        """
            Returns the num of intersections, fills self.checkList
        """

        if (checkAll):
            px = np.zeros([2])

        seg_r = np.array([(p1[0] - p0[0]) / 2.0, (p1[1] - p0[1]) / 2.0])
        seg_c = p0[:2] + seg_r
        segLen = 2 * norm(seg_r)

        lines_seg_r = np.zeros([2])

        L0 = np.zeros([2]) #axis along segment normal
        L1 = np.zeros([2]) #axis along lines segment normal

        if (seg_r[0] != 0.0):
            L0[0] = seg_r[1] / seg_r[0]
            L0[1] = -1.0
        else:
            L0[0] = -1.0
            L0[1] = 0.0

        T = np.zeros([2])

        while seg_from < seg_to:

            p3, shift = self.GetPnt(seg_from + 1)
            if (ignoreGaps and shift != 0):
                seg_from += (shift + 1)
                continue
            p2, _ = self.GetPnt(seg_from)

            lines_seg_r[0] = (p3[0] - p2[0]) / 2.0
            lines_seg_r[1] = (p3[1] - p2[1]) / 2.0

            T[0] = p2[0] + lines_seg_r[0] - seg_c[0]
            T[1] = p2[1] + lines_seg_r[1] - seg_c[1] 

        #L0 along segment normal

            abs_T_L = abs(np.dot(T, L0))
            abs_lines_seg_r_L = abs(np.dot(lines_seg_r, L0))
            # abs_seg_r_L = 0.0

            if (abs_T_L > abs_lines_seg_r_L):
                seg_from += (shift + 1)
                continue

        #L1 along lines_segment normal

            if (lines_seg_r[0] != 0.0):
                L1[0] = lines_seg_r[1] / lines_seg_r[0]
                L1[1] = -1.0
            else:
                L1[0] = -1.0
                L1[1] = 0.0

            abs_T_L = abs(np.dot(T, L1))
            # abs_lines_seg_r_L = 0.0
            abs_seg_r_L = abs(np.dot(seg_r, L1))

            if (abs_T_L > abs_seg_r_L):
                seg_from += (shift + 1)
                continue

            if (not checkAll):
                return 1    #are intersect

            if (L0[1] != 0.0):
                k1 = -L0[0] / L0[1] #line 1
                if (L1[1] != 0.0):
                    k2 = -L1[0] / L1[1]
                    b1 = p0[1] - k1 * p0[0]
                    b2 = p2[1] - k2 * p2[0]
                    px[0] = (b2 - b1) / (k1 - k2)
                    px[1] = k1 * px[0] + b1
                else:
                    px[0] = self.GetPnt(seg_from)[0][0]
                    px[1] = k1 * (p2[0] - p0[0]) + p0[1]
            else:
                px[0] = p0[0] #b1
                px[1] = -L1[0] / L1[1] * (p0[0] - p2[0]) + p2[1] #k2 * b1 + b2, for b2 look double else


            self.checkList[0, num] = seg_from
            self.checkList[1, num] = norm(px - p0[:2]) / segLen
            self.checkList[2, num] = (shift != 0)
        
            num += 1
            seg_from += (shift + 1)
        
        return num

    def Check_segment_intersections00(self, p1 : np.ndarray, seg_from : int, seg_to : int, checkAll : bool = False, num : int = 0, ignoreGaps : bool = False):
        """
            Checks intersections with segment = p1 - [0.0, 0.0]
            Returns the num of intersections, fills self.checkList
        """

        if (checkAll):
            px = np.zeros([2])

        seg_r = p1[:2] / 2.0
        segLen = 2 * norm(seg_r)

        lines_seg_r = np.zeros([2])

        L0 = np.zeros([2]) #axis along segment normal
        L1 = np.zeros([2]) #axis along lines segment normal

        if (seg_r[0] != 0.0):
            L0[0] = seg_r[1] / seg_r[0]
            L0[1] = -1.0
        else:
            L0[0] = -1.0
            L0[1] = 0.0

        T = np.zeros([2])

        while seg_from < seg_to:

            p3, shift = self.GetPnt(seg_from + 1)
            if (ignoreGaps and shift != 0):
                seg_from += (shift + 1)
                continue
            p2, _ = self.GetPnt(seg_from)

            lines_seg_r[0] = (p3[0] - p2[0]) / 2.0
            lines_seg_r[1] = (p3[1] - p2[1]) / 2.0

            T[0] = p2[0] + lines_seg_r[0] - seg_r[0]
            T[1] = p2[1] + lines_seg_r[1] - seg_r[1] 

        #L0 along segment normal

            abs_T_L = abs(np.dot(T, L0))
            abs_lines_seg_r_L = abs(np.dot(lines_seg_r, L0))

            if (abs_T_L > abs_lines_seg_r_L):
                seg_from += (shift + 1)
                continue

        #L1 along lines_segment normal

            if (lines_seg_r[0] != 0.0):
                L1[0] = lines_seg_r[1] / lines_seg_r[0]
                L1[1] = -1.0
            else:
                L1[0] = -1.0
                L1[1] = 0.0

            abs_T_L = abs(np.dot(T, L1))
            abs_seg_r_L = abs(np.dot(seg_r, L1))

            if (abs_T_L > abs_seg_r_L):
                seg_from += (shift + 1)
                continue

            if (not checkAll):
                return 1    #are intersect

            if (L0[1] != 0.0):
                k1 = -L0[0] / L0[1] #line 1
                if (L1[1] != 0.0):
                    k2 = -L1[0] / L1[1]
                    px[0] = (p2[1] - k2 * p2[0]) / (k1 - k2)
                    px[1] = k1 * px[0]
                else:
                    px[0] = self.GetPnt(seg_from)[0][0]
                    px[1] = k1 * p2[0]
            else:
                px[0] = 0.0 #b1
                px[1] = L1[0] / L1[1] * p2[0] + p2[1] #k2 * b1 + b2, for b2 look double else

            self.checkList[0, num] = seg_from
            self.checkList[1, num] = norm(px) / segLen
            self.checkList[2, num] = (shift != 0)
        
            num += 1
            seg_from += (shift + 1)
        
        return num

    def Check_if_obb_intersection(self, obb_L2G : np.ndarray, obb_half_length, obb_half_width, seg_from : int, seg_to : int):
        """
            obb - is oriented bounding box on the plane, obb_l2G is 3x3 dimensions.\n
            Returns the first intersected segment index, or -1 if there is no intersections found.
            Gaps are ignored
        """

        #FrontUp radius
        obb_r1 = np.array([obb_L2G[0, 0] * obb_half_length + obb_L2G[0, 1] * obb_half_width, obb_L2G[1, 0] * obb_half_length + obb_L2G[1, 1] * obb_half_width])
        #FrontDown radius
        obb_r2 = np.array([obb_L2G[0, 0] * obb_half_length - obb_L2G[0, 1] * obb_half_width, obb_L2G[1, 0] * obb_half_length - obb_L2G[1, 1] * obb_half_width])
        #OBB is symmetric, so here are only 2 radii out of 4

        lines_seg_r = np.zeros([2])

        L0 = np.zeros([2]) #axis normal to segment
        L1 = np.zeros([2]) #axis along length of obb
        L2 = np.zeros([2]) #axis along width of obb
        L1[:] = obb_L2G[:2, 0]
        L2[:] = obb_L2G[:2, 1]

        T = np.zeros([2])

        while seg_from < seg_to:

            p1, shift = self.GetPnt(seg_from + 1)
            if (shift != 0):
                seg_from += (shift + 1)
                continue
            p0, _ = self.GetPnt(seg_from)

            lines_seg_r[0] = (p1[0] - p0[0]) / 2.0
            lines_seg_r[1] = (p1[1] - p0[1]) / 2.0

            T[0] = p0[0] + lines_seg_r[0] - obb_L2G[0, 2] 
            T[1] = p0[1] + lines_seg_r[1] - obb_L2G[1, 2] 

        #L0 along segment

            if (lines_seg_r[0] != 0.0):
                L0[0] = lines_seg_r[1] / lines_seg_r[0]
                L0[1] = -1.0
            else:
                L0[0] = -1.0
                L0[1] = 0

            abs_T_L = abs(np.dot(T, L0))
            # abs_lines_seg_r_L = 0.0
            abs_obb_r1_L = abs(np.dot(obb_r1, L0))
            abs_obb_r2_L = abs(np.dot(obb_r2, L0))

            if (abs_obb_r1_L >= abs_obb_r2_L):
                if (abs_T_L > abs_obb_r1_L):
                    seg_from += 1
                    continue
            elif (abs_T_L > abs_obb_r2_L):
                seg_from += 1
                continue

        #L1 along length of obb

            abs_T_L = abs(np.dot(T, L1))
            abs_lines_seg_r_L = abs(np.dot(lines_seg_r, L1))
            abs_obb_r1_L = abs(np.dot(obb_r1, L1))
            abs_obb_r2_L = abs(np.dot(obb_r2, L1))

            if (abs_obb_r1_L >= abs_obb_r2_L):
                if (abs_T_L > (abs_obb_r1_L + abs_lines_seg_r_L)):
                    seg_from += 1
                    continue
            elif (abs_T_L > (abs_obb_r2_L + abs_lines_seg_r_L)):
                seg_from += 1
                continue

        #L2 along width of obb

            abs_T_L = abs(np.dot(T, L2))
            abs_lines_seg_r_L = abs(np.dot(lines_seg_r, L2))
            abs_obb_r1_L = abs(np.dot(obb_r1, L2))
            abs_obb_r2_L = abs(np.dot(obb_r2, L2))

            if (abs_obb_r1_L >= abs_obb_r2_L):
                if (abs_T_L > (abs_obb_r1_L + abs_lines_seg_r_L)):
                    seg_from += 1
                    continue
            elif (abs_T_L > (abs_obb_r2_L + abs_lines_seg_r_L)):
                seg_from += 1
                continue

            return seg_from    #intersecting with segment vx_from .. vx_from + 1 and maybe others next
        
        return -1   #non-intersecting at all

N = 400
edgeShift = 3

polyline = Polyline(N, edgeShift)

#пример использования:
from Lidar import Lidar

lidar = Lidar.Create(0)
lidar.Start()

lidStatus = lidar.GetLinesXY(polyline) #это синхронизирующий вызов, делающийся всегд из python

Nlines = polyline.Nlines
edgeGaps = (polyline.gapsIdxs[0], polyline.gapsIdxs[polyline.Ngaps - 1])
seg_from = polyline.edgeShift
seg_to = Nlines - seg_from - 1

rLC = np.array([1.0, 0.0]) #некоторая точка (в нашем случае целевая)

xFar_i_reserve, xFar_pnt_reserve, goal2LineDist = polyline.Closest_pnt(rLC, 0, Nlines)

xNear_i_reserve, xNear_pnt_reserve, minDist = polyline.Closest_pnt00(seg_from, seg_to)

xNum = polyline.Check_segment_intersections00(rLC, 0, Nlines, checkAll = True)

#здесь будет вызов polylineCpp.synchronize(polyline.cppID)

if (xNum != 0):

    i_ = polyline.checkList[1, :xNum].argmin()
    xGoalNear_i = int(polyline.checkList[0, i_])
    xGoalNear_pnt = polyline.checkList[1, i_] * rLC
    xGoalNear_isGap = (polyline.checkList[2, i_] != 0.0)

    i_ = polyline.checkList[1, :xNum].argmax()
    xGoalFar_i = int(polyline.checkList[0, i_])
    xGoalFar_pnt = polyline.checkList[1, i_] * rLC
    xGoalFar_isGap = (polyline.checkList[2, i_] != 0.0)

    numOfxGaps = int(np.sum(polyline.checkList[2, :xNum]))
    if (numOfxGaps == xNum):
        goInGapCheck = True
    else:
        goInGapCheck = ((xNum % 2) == 1)
    pureFreeWay = False

else:

    xGoalNear_i = xNear_i_reserve
    xGoalNear_pnt = xNear_pnt_reserve
    xGoalFar_isGap = False
    
    xGoalFar_i = xFar_i_reserve
    xGoalFar_pnt = xFar_pnt_reserve
    xGoalFar_isGap = False

    numOfxGaps = 0
    goInGapCheck = False
    pureFreeWay = True