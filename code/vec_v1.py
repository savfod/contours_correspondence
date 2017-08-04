import math


class Vec:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2)

    def coeff_prod(self, coeff):
        return Vec(self.x * coeff, self.y * coeff)

    def __sub__(self, other):
        return self + other.coeff_prod(-1)

    def __mul__(self, other):
        # other is expected to be num. num*vec won't work
        return Vec(self.x * other, self.y * other)

    def __truediv__(self, other):
        # other is expected to be num. num*vec won't work
        return Vec(self.x / other, self.y / other)

    def __div__(self, other):
        # other is expected to be num. num*vec won't work
        return Vec(self.x / other, self.y / other)

    def __str__(self):
        return str(self.x) + " " + str(self.y)


class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def coeff_prod(self, coeff):
        return Vec3(self.x * coeff, self.y * coeff, self.z * coeff)

    def __sub__(self, other):
        return self + other.coeff_prod(-1)

    def __mul__(self, other):
        # other is expected to be num. num*vec won't work
        return self.coeff_prod(other)

    def __truediv__(self, other):
        # other is expected to be num. num*vec won't work
        return self.coeff_prod(1 / other)

    def __div__(self, other):
        # other is expected to be num. num*vec won't work
        return self.coeff_prod(1 / other)

    def __str__(self):
        return str(self.x) + " " + str(self.y) + " " + str(self.z)
