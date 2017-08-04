import math

def vec_prod(v1, v2):
    return v1.x*v2.y - v1.y*v2.x

def sc_prod(v1, v2):
    return v1.x*v2.x + v1.y*v2.y

def find_lowest_point_index(points):
    lowest_i = 0
    lowest_point = points[0]
    for i in range(len(points)):
        if points[i].y < lowest_point.y:
            lowest_i = i
            lowest_point = points[i]
    return lowest_i

def normalize_clockwise(points):
    sum_prods = 0
    points_copy = points
    for p1, p2, p3 in zip(points_copy, points_copy[1:], points_copy[2:]):
        v1 = p2 - p1
        v2 = p3 - p2
        # print(str(v1), str(v2), str(p1), str(p2))

        if sc_prod(v1, v2):
            cos_angle = sc_prod(v1, v2) / (abs(v1)*abs(v2))
            cos_angle = max(min(cos_angle, 1.), -1.)
            # print(cos_angle)
            angle = math.acos(cos_angle)
            angle = math.copysign(angle, vec_prod(v1, v2))
            sum_prods += angle

    if sum_prods < 0:
        points = points[::-1]

    return points




def get_convex_hull(points):
    #start with lowest point
    low_i = find_lowest_point_index(points)

    points = points[low_i:] + points[:low_i]

    convex_hull = [points[0]]

    points = points[1:] + points[:1]
    for p in points:
        convex_hull.append(p)

        tested = False
        while len(convex_hull) >= 3 and not tested:
            v1 = convex_hull[-2] - convex_hull[-1]
            v2 = convex_hull[-2] - convex_hull[-3]
            # print(v1, v2, vec_prod(v1, v2))
            #assert anti-clock-wise point seq
            if vec_prod(v1, v2) < 0:
                convex_hull.pop(-2)
            else:
                tested = True
    convex_hull = convex_hull[:-1] #remove first-last duplicate
    return convex_hull


# def calc_wurfs_points(points, args):
#     for i in range(args.smooth):
#         points = smooth_points(points)
#     points = filter_big_angle(points, args.filter_angle)
#     points = length_points_filter(points, new_count=200)
#     print(len(points))
#     skip = len(points) // 5

#     wurfs_points = []
#     if len(points) >= 1 + 4*skip:
#         for i in range(len(points)):
#             shifted = points[i:] + points[:i]
#             five_points = shifted[:5*skip:skip]
#             x, y = calc_wurfs(five_points)
#             wurfs_points.append(Vec(x, y) / 2)
#     print(len(wurfs_points))
#     return wurfs_points
