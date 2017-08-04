from vec_v1 import Vec


def vec_prod(v1, v2):
    return v1.x*v2.y - v1.y*v2.x

def line_intersection(p1, v1, p2, v2, default=Vec(0, 0)):
    # hard-coded formula for line intersection. Write-only
    # det = v1.y*v2.x - v2.y*v1.x
    det = -vec_prod(v1, v2)
    right = p2 - p1
    v1coeff = -vec_prod(right, v2)

    if (det == 0):
        return default

    return p1 + v1.coeff_prod(v1coeff / det)


class Line:
    def __init__(self, start_point, vec):
        self.start = start_point
        self.vec = vec

    def intersect(self, other):
        return line_intersection(self.start, self.vec, other.start, other.vec)

    def diff_side_of_line(self, p1, p2):
        prod1 = vec_prod(p1 - self.start, self.vec)
        prod2 = vec_prod(p2 - self.start, self.vec)
        return prod1 * prod2 <= 0
