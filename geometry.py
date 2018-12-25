import numpy as np

def norm(v):
    return np.sqrt(v.dot(v))

def normalize(v):
    return v/norm(v)

def rotate(v, d):
    # rotate self by an angle given by the direction d
    # neutral rotation is top (ie (0,1))
    d = normalize(d)
    return np.array([v[0] * d[1] + v[1] * d[0], v[1] * d[1] - v[0] * d[0]])


def rotate90(v):
    # orthogonal dans le sens direct
    return np.array([-v[1], v[0]])


def direct_order(v0, v1, v2):
    if v2.dot(rotate90(v0)) >= 0:
        return v1.dot(rotate90(v0)) >= 0 and v1.dot(rotate90(v2)) <= 0
    else:
        return not(direct_order(v2, v1, v0))

# point
def is_in_quad(p, quad):
    # checks if point is in quadrilatere
    qp = [quad[1, 0], quad[0, 0], quad[0, 1], quad[1, 1], quad[1, 0], quad[0, 0]]
    checks = [direct_order(qp[i-1] - qp[i], p - qp[i], qp[i+1] - qp[i]) for i in range(1, 5)]
    return all(checks)


def intersection_distance(p, v, s1, s2):
    rs = rotate90(s2 - s1)
    rv = rotate90(v)
    return (s1 - p).dot(rs) / v.dot(rs), (p - s1).dot(rv)/(s2-s1).dot(rv)


def angle(v1, v2):
    a = v2 - v1
    theta = np.arctan(a[1]/a[0])
    if a[0] > 0:
        if a[1] > 0:
            return theta
        return 2 * np.pi + theta
    return np.pi + theta


def cut(v, M):
    return max(min(v, M), - M)

