from drawer_v1 import Drawer
from vec_v1 import Vec
from wurfs_methods import normalize_clockwise, get_convex_hull

def init_tk_drawer():
    drawer = Drawer()
    tk = drawer.tk

    def zoom(event):
        print("Hello windows/macos! Not-tested scaling.")
        drawer.scale(1.1 ** event.delta, event.x, event.y)

    def zoom_in(event):
        drawer.scale(1.1, event.x, event.y)

    def zoom_out(event):
        drawer.scale(1.1 ** (-1), event.x, event.y)

    tk.bind("<MouseWheel>", zoom)
    tk.bind("<Button-4>", zoom_in)
    tk.bind("<Button-5>", zoom_out)

    return drawer.tk, drawer

def find_lowest_point_index(points):
    lowest_i = 0
    lowest_point = points[0]
    for i in range(len(points)):
        if points[i].y < lowest_point.y:
            lowest_i = i
            lowest_point = points[i]
    return lowest_i


def length_points_filter(points, start=0, new_count=100):
    low_i = find_lowest_point_index(points)
    points = points[low_i:] + points[:low_i]

    sum_length = sum(abs(p2-p1) for p1,p2 in zip(points, points[1:]))
    part_length = sum_length/new_count
    # print(part_length)

    result_points = []
    cur_length = 0
    for p1, p2 in zip(points, points[1:] + points[:1]):
        vec = p2 - p1

        next_length = cur_length + abs(vec)

        for i in range(int((next_length // part_length) - (cur_length // part_length))):
           precise_length = ((cur_length // part_length) + i + 1) * part_length

           precise_vec = vec * (precise_length - cur_length) / abs(vec)
           new_point = p1 + precise_vec
           result_points.append(new_point)

        cur_length += abs(vec)
    return result_points


WIDTH = 1000
HEIGHT = 1000
def read_points(filename):
    points = []

    with open(filename) as f:
        lines = f.readlines()

        size_str_skiped = False
        for l in lines:
            l = l.strip()

            if l.startswith("#") or len(l) == 0:
                continue
            if filename.endswith(".frommat") and not size_str_skiped:
                size_str_skiped = True
                continue

            y, x = map(float, l.split()) #rev
            x /= WIDTH
            y /= HEIGHT
            y = 1 - y
            points.append(Vec(x, y))

    return points


def write_points(filename, points):
    with open(filename, 'w') as f:
        for p in points:
            y, x = p.x, p.y #rev
            x = 1 - x
            x = int(WIDTH*x)
            y = int(HEIGHT*y)
            f.write("{0} {1}\n".format(x, y))




def smooth_points(points):
    return [(p1+p2)/2 for p1, p2 in zip(points, points[1:] + points[:1])]


def filter_big_angle(points, cos_value = 0.8):
    def cos_angle(vec1, vec2):
        return (vec1.x*vec2.x + vec1.y*vec2.y)/(abs(vec1) * abs(vec2))

    points = points + points[:2]
    filtered_points = []
    for (p1, p2, p3) in zip(points, points[1:], points[2:]):
        v1, v2 = p2-p1, p3-p2

        if abs(v1)*abs(v2) > 0 and cos_angle(v1, v2) < cos_value:
            filtered_points.append(p2)

    return filtered_points


def add_prepare_args(parser):
    # parser.add_argument('-f', '--files', type=str, nargs="+", required=True, help='input file name')
    # parser.add_argument('-f2', '--files_other', type=str, nargs="+", help='input file name for other files to calc diff')
    # parser.add_argument('-n', '--no_image', action="store_true", help='not to draw image')
    parser.add_argument('-s', '--smooth', type=int, default="0", help='how many points to use')
    parser.add_argument('-a', '--filter_angle', type=float, help='remove points with big cos_angle')
    parser.add_argument('-c', '--use_convex_hull', action="store_true", help='use convex hull instead of points')
    parser.add_argument('-lfp', '--length_filter_points', default=200, type=int, help='length_filter points count (0 to disable)')


def prepare_points(points, args):
    points = normalize_clockwise(points)
    for i in range(args.smooth):
        points = smooth_points(points)

    # points = normalize_clockwise(points)
    if args.use_convex_hull:
        points = get_convex_hull(points)

    if args.filter_angle:
        points = filter_big_angle(points, args.filter_angle)

    if args.length_filter_points >= 1:
        points = length_points_filter(points, new_count=args.length_filter_points)


    return points
