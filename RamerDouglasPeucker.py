import numpy as np
import math
import scipy


def DouglasPeucker(PointList, epsilon):
    """Returns reduced points list using the Ramer–Douglas–Peucker algorithm

        Inputs:
        PointList: list of (x,y) tuples
        epsilon
    """
    pointArray = np.array(PointList)
    dmax =0
    index = 0
    end = pointArray.shape[0]

    p1=np.array(pointArray[0])
    p2=np.array(pointArray[-1])

    for i in range(0, end, 2):
        p3 = np.array(pointArray[i])
        d  = abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
        if d > dmax:
            index = i
            dmax = d

    results = []

    if dmax > epsilon:
        recResults1 = DouglasPeucker(pointArray[:index], epsilon)
        recResults2 = DouglasPeucker(pointArray[index:end], epsilon)
        results.extend(recResults1)
        results.extend(recResults2)
    else:
        results = [(p1[0], p1[1]), (p2[0], p2[1])]

    return results
