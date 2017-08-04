from drawer_v1 import Drawer
from vec_v1 import Vec, Vec3
from line_v1 import Line, vec_prod
from utils import read_points, prepare_points, add_prepare_args, init_tk_drawer

import argparse
import math

DESCRIPTION = '''
Program to draw wurfs for contours
'''


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=DESCRIPTION,
    )
    # parser.add_argument('--rounds', type=int, default=2, help='how many rounds each pair plays')
    parser.add_argument('-f', '--files', type=str, nargs="+", required=True, help='input file name')
    parser.add_argument('-f2', '--files_other', type=str, nargs="+", help='input file name for other files to calc diff')
    parser.add_argument('-n', '--no_image', action="store_true", help='not to draw image')
    parser.add_argument('-dps', '--diff_points_share', type=float, default=0.4, help='share of points to use in diff')
    parser.add_argument('-wm', '--wurfs_method', default=1, help='index of wurfs_method to use')
    parser.add_argument('-dm', '--diff_method', default=1, help='index of diff method to use')
    parser.add_argument('-nl', '--normalize_length', action="store_true", default=False, help='use length normalizing')
    parser.add_argument('-ws', '--wurfs_skip', type=int, help='count of points to skip in wurfs')
    parser.add_argument('-um', '--use_metrics', type=int, default=0, help='use metrics')


    add_prepare_args(parser)

    # parser.add_argument('--points_multiplier', type=int, default="2", help='how many points to use')
    # parser.add_argument('--tangent_curve', action="store_true", help='draw tangent curve')
    # parser.add_argument('--points_count', type=int, default=180, help='how many points to use (more points is slower)')
    # # parser.add_argument('--cyclic', action="store_true", default="False", help='draw tangent curve')
    # # parser.add_argument('--draw_points', action="store_true", default=False, help='draw selected points')
    # parser.add_argument('--draw_points', nargs="+", help='draw selected points. format: x_coordinate,label[,draw_tangent]')

    parsed_args = parser.parse_args()
    return parsed_args


def vec_div(v1, v2):
    # collinear vecs only
    if abs(v1.x) >= abs(v1.y):
        return v1.x / v2.x
    return v1.y / v2.y


def wurf(p1, p2, p3, p4):
    # collinear vecs only
    return vec_div(p3 - p1, p3 - p2) / vec_div(p4 - p1, p4 - p2)


def calc_wurfs(five_points):
    def calc_left_wurf(five_points):
        p1, p2, p3, p4, p5 = five_points

        p14_25 = Line(p1, p4).intersect(Line(p2, p5))
        p14_35 = Line(p1, p4).intersect(Line(p3, p5))

        return wurf(p1, p4, p14_25, p14_35)

    return calc_left_wurf(five_points), calc_left_wurf(five_points[::-1])


def get_colour(index):
    colors = ['blue', 'green', 'red', 'pink']
    return colors[index % len(colors)]


def calc_average(points):
    s = Vec(0, 0)
    for p in points:
        s += p
    return s / len(points)


def calc_perimeter(points, cyclic=True):
    if len(points) < 2:
        return 0
    length = sum(abs(p2 - p1) for (p1, p2) in zip(points, points[1:]))
    if cyclic:
        length += abs(points[-1] - points[0])
    return length


def calc_complex_correlation(points1, points2):
    zipped = list(zip(points1, points2))
    if len(points1) == len(points2):
        zipped.append((points1[0], points2[0]))

    corr = complex(0)
    for (p11, p21), (p12, p22) in zip(zipped, zipped[1:]):
        def c_diff(v1, v2):
            d = v2 - v1
            return complex(d.x, d.y)

        c1 = c_diff(p11, p12)
        c2 = c_diff(p21, p22)
        corr += c1 * c2.conjugate()

    return corr


def toVec(ar):
    assert(len(ar) == 2)
    return Vec(ar[0], ar[1])


def toVec3(ar):
    assert(len(ar) == 3)
    return Vec(ar[0], ar[1], ar[2])


def cyclic_shifts(points):
    cyclic_shifts = []
    for i in range(len(points)):
        cyclic_shifts.append(points[i:] + points[:i])
    return cyclic_shifts


def prepare_and_calc_wurfs_points(points, args):
    points = prepare_points(points, args)

    inv_points = []

    # if args.wurfs_method in ["201"]:
    if args.wurfs_method in ["12"]:
        skip = len(points) // 5
        if args.wurfs_skip:
            skip = args.wurfs_skip


        if len(points) >= 1 + 4*skip:
            for shifted in cyclic_shifts(points):
                five_points = shifted[:5*skip:skip]
                x, y = calc_wurfs(five_points)
                inv_points.append(Vec(x, y) / 2)

    # elif args.wurfs_method in ["202"]:
    elif args.wurfs_method in ["13"]:
        skip = len(points) // 4
        if len(points) >= 1 + 3*skip:
            for shifted in cyclic_shifts(points):

                def get_line(ind):
                    next = (ind + 1)%len(shifted)
                    return Line(shifted[ind], shifted[next] - shifted[ind-1])

                try:
                    p3 = shifted[0]
                    l1 = get_line(0)
                    l2 = get_line(skip)
                    ipt1 = shifted[skip]
                    l3 = get_line(2*skip)
                    l4 = get_line(3*skip)
                    ipt2 = shifted[2*skip]
                    ipt3 = shifted[3*skip]

                    p2 = l1.intersect(l2)
                    p1 = l2.intersect(l3)
                    p5 = l3.intersect(l4)
                    p4 = l4.intersect(l1)

                    # x, y = calc_wurfs([p1,p2,p3,p4,p5])
                    x, y = calc_wurfs([p2, ipt1, ipt2, ipt3 ,p4])
                    inv_points.append(Vec(x,y) / 2)
                except:
                    pass

    # elif args.wurfs_method in ["203"]:
    elif args.wurfs_method in ["5", "TR"]:
        # euclidian invariant, triangle lengths
        norm_len = calc_perimeter(points)
        skip = len(points) // 3
        if args.wurfs_skip:
            skip = args.wurfs_skip

        if len(points) >= 1 + 2*skip:
            for shifted in cyclic_shifts(points):
                x, y, z = shifted[:3*skip:skip]
                if args.normalize_length:
                    inv_points.append(Vec3(abs(x-y), abs(x-z), abs(y-z)) / norm_len)
                else:
                    inv_points.append(Vec3(abs(x-y), abs(x-z), abs(y-z)))

    # elif args.wurfs_method in ["204"]:
    elif args.wurfs_method in ["6", "CR1"]:
        # euclidian invarian
        skip = len(points) // 7
        if args.wurfs_skip:
            skip = args.wurfs_skip

        if len(points) >= 1 + 6*skip:
            norm_len = calc_perimeter(points)

            for shifted in cyclic_shifts(points):
                d = []
                for sk in 0.5*skip, 1*skip, 2*skip:
                    sk = int(sk)
                    d.append(abs(shifted[sk] - shifted[-sk])*skip/sk)

                if args.normalize_length:
                    inv_points.append(Vec3(d[0], d[1], d[2]) / norm_len)
                else:
                    inv_points.append(Vec3(d[0], d[1], d[2]))

    # elif args.wurfs_method in ["205"]:
    elif args.wurfs_method in ["7", "CR2"]:
        #euclidian invarian
        skip = len(points) // 4
        if args.wurfs_skip:
            skip = args.wurfs_skip

        if len(points) >= 1 + 3*skip:
            norm_len = calc_perimeter(points)

            for shifted in cyclic_shifts(points):
                d = []
                d.append(abs(shifted[0] - shifted[2*skip]))
                d.append(abs(shifted[skip] - shifted[-skip]))
                if args.normalize_length:
                    inv_points.append(Vec(d[0], d[1]) / norm_len)
                else:
                    inv_points.append(Vec(d[0], d[1]))

    # elif args.wurfs_method in ["206"]:
    elif args.wurfs_method in ["1", "NO"]:
        # place of points :)
        inv_points = points[:]

    # elif args.wurfs_method in ["207"]:
    elif args.wurfs_method in ["8", "CUR"]:
        # curvative and curvative derivative
        skip = len(points) // 10
        if args.wurfs_skip:
            skip = args.wurfs_skip

        if len(points) >= 1 + 4*skip:
            norm_len = calc_perimeter(points)

            for shifted in cyclic_shifts(points):
                curvative = (shifted[skip] - shifted[0]) - (shifted[0] - shifted[-skip])
                curvative_next = (shifted[2*skip] - shifted[skip]) - (shifted[skip] - shifted[0])

                d = []
                d.append(abs(vec_prod(curvative, (shifted[skip] - shifted[0]))))
                d.append(abs(vec_prod(curvative_next - curvative, (shifted[skip] - shifted[0])))*abs(shifted[skip] - shifted[0]))
                # inv_points.append(Vec3(d[0], d[1], d[2]))
                # inv_points.append(Vec3(d[0], d[1], d[2]) / norm_len)
                if args.normalize_length:
                    inv_points.append(Vec(d[0], d[1]) * 1000 / norm_len)
                else:
                    inv_points.append(Vec(d[0], d[1]))

    # elif args.wurfs_method in ["208"]:
    elif args.wurfs_method in ["3", "TAN"]:
        # tangent angle
        skip = 1
        if args.wurfs_skip:
            skip = args.wurfs_skip

        for p1, p2 in zip(points, points[1:] + points[:1]):
            inv_points.append((p2-p1)/abs(p2-p1))

    # elif args.wurfs_method in ["209"]:
    elif args.wurfs_method in ["2", "MASS"]:
        # center of mass
        av = calc_average(points)
        inv_points += [av] * 5
        # following code can break on single value
        # repeating doesn't change result metrics


    # elif args.wurfs_method in ["2010"]:
    elif args.wurfs_method in ["4", "M1"]:
        # normalized average
        av = calc_average(points)
        for p in points:
            inv_points.append(p - av)

    # elif args.wurfs_method in ["2011"]:
    elif args.wurfs_method in ["10", "M1.5"]:
        # normalized average and (av_x**2) and (av_y**2)
        av = calc_average(points)
        norm_av_points = [p - av for p in points]

        def av_sq(nums):
            return math.sqrt(sum([n**2 for n in nums])/len(nums))
        av_x_squared = av_sq([p.x for p in norm_av_points])
        av_y_squared = av_sq([p.y for p in norm_av_points])

        for p in norm_av_points:
            inv_points.append(Vec(p.x/av_x_squared, p.y/av_y_squared))

    # elif args.wurfs_method in ["2012"]:
    elif args.wurfs_method in ["11", "M2"]:
        # normalized average and second momentum. ortomatrix invariant
        av = calc_average(points)
        norm_av_points = [p - av for p in points]

        def av_num(nums):
            return sum(nums) / len(nums)
        av_xx = av_num([p.x**2 for p in norm_av_points])
        av_xy = av_num([p.x*p.y for p in norm_av_points])
        av_yy = av_num([p.y**2 for p in norm_av_points])
        # print(av_xx, av_xy, av_yy)

        v1 = av_xx - av_yy
        v2 = av_xy * 2
        if abs(v1) > 10**-6:
            #rotate
            two_alpha = math.atan(v2/v1)
            alpha = two_alpha/2
            s = math.sin(alpha)
            c = math.cos(alpha)

            # print(sum( (c*p.x + s*p.y)*(-s*p.x + c*p.y) for p in norm_av_points))
            norm_av_points = [Vec(c*p.x + s*p.y, -s*p.x + c*p.y) for p in norm_av_points]

        av_xx = av_num([p.x**2 for p in norm_av_points])
        av_xy = av_num([p.x*p.y for p in norm_av_points])
        av_yy = av_num([p.y**2 for p in norm_av_points])
        # print(av_xx, av_xy, av_yy)

        norm_av_points = [Vec(p.x/math.sqrt(av_xx), p.y/math.sqrt(av_yy)) for p in norm_av_points]
        # norm_av_points = prepare_points(norm_av_points, args)

        for p in norm_av_points:
            inv_points.append(p)

    # elif args.wurfs_method in ["2013"]:
    elif args.wurfs_method in ["9", "ACOR"]:
        # autocorr:
        diffs = [(p2 - p1) for p1, p2 in zip(points, points[1:] + points[:1])]
        diffs = [complex(d.x, d.y) for d in diffs]

        # if args.sqrt_len_optimization:
        diffs = [d/(abs(d)**0.5) for d in diffs]

        autocorrelations = []
        for diffs_shifted in cyclic_shifts(diffs):
            value = 0
            for d1, d2 in zip(diffs, diffs_shifted):
                value += d1 * d2.conjugate()

            autocorrelations.append(value)

        #norming
        coef = abs(autocorrelations[0])
        for v in autocorrelations:
            inv_points.append(Vec(v.real, v.imag)/coef)


    else:
        raise IOError("Unexpected method")

    # print(len(inv_points))
    return inv_points



def calc_and_draw_values(drawer, files, args, fill='green'):
    values = []
    for file_index, filename in enumerate(files):
        points = read_points(filename)
        wurfs_points = prepare_and_calc_wurfs_points(points, args)
        values.append(wurfs_points)

        if drawer is not None:
            prev_vec = Vec(0, 0)
            for vec in wurfs_points:
                drawer.draw_circle(vec, fill=fill)
                drawer.draw_line(prev_vec, vec, fill=fill)
                # drawer.draw_line(prev_vec, vec, fill=get_colour(file_index))
                prev_vec = vec

    return values


def calc_diff(wurfs1, wurfs2, args):
    # if args.diff_method in ["301"]: # old numeration
    if args.diff_method in ["6", "AVMIN"]:
        distances = []
        for i in range(len(wurfs1)):

            i_distances = []
            for j in range(len(wurfs2)):
                i_distances.append(abs(wurfs1[i] - wurfs2[j]))
            distances.append(min(i_distances))
        distances.sort()
        used_diff_count = int(len(wurfs1) * args.diff_points_share)
        distances_part = distances[:used_diff_count]

        return 1000 * (sum(distances_part) / len(distances_part))
        # return 1000 * (sum(distances_part) / len(distances_part) - sum(distances)/len(distances)/100)

    # elif args.diff_method in ["302"]:
    elif args.diff_method in ["7", "AVC"]:
        dists = []
        for shifted in cyclic_shifts(wurfs2):
            dist = 0

            diff_pt_count = int(min(len(wurfs1), len(wurfs2)) * args.diff_points_share)
            for j in range(diff_pt_count // 2):
                dist += abs(wurfs1[j] - shifted[j])
                dist += abs(wurfs1[-j] - shifted[-j])

            dists.append(dist)

        return min(dists)

    # elif args.diff_method in ["303"]:
    elif args.diff_method in ["1", "AV"]:

        #trivial
        zipped = list(zip(wurfs1, wurfs2))
        used_diff_count = int(len(zipped) * args.diff_points_share)

        dist = 0
        for w1, w2 in zipped[:used_diff_count]:
            dist += abs(w2 - w1)
        return dist

    # elif args.diff_method in ["304"]:
    elif args.diff_method in ["2", "DYN"]:
        # dynamic
        # not really correct with wurfs1[0] ~ wurfs2[0]
        path_dist = [[-1]*len(wurfs2) for i in range(len(wurfs1))]

        def w_dist(i, j):
            return abs(wurfs1[i] - wurfs2[j])

        for j in range(len(wurfs2)):
            path_dist[0][j] = w_dist(0, j)
        for i in range(1, len(wurfs1)):
            path_dist[i][0] = path_dist[i-1][0] + w_dist(i, 0)

        for i in range(1, len(wurfs1)):
            for j in range(1, len(wurfs2)):
                prev_i = path_dist[i-1][j]
                prev_j = path_dist[i][j-1] - w_dist(i, j-1)  # NB
                path_dist[i][j] = min(prev_i, prev_j) + w_dist(i, j)

        return min(path_dist[-1])

    # elif args.diff_method in ["305"]:
    elif args.diff_method in ["3", "DYN2"]:
        # dynamic with sqrt product
        # (idea from "computable elastic distances between shapes")
        # not really correct with wurfs1[0] ~ wurfs2[0]
        path_dist = [[-1]*len(wurfs2) for i in range(len(wurfs1))]
        prevs = [[None]*len(wurfs2) for i in range(len(wurfs1))]

        def w_dist(i, j, prev_i_j):
            dist = abs(wurfs1[i] - wurfs2[j])

            # for elasctic distance article. With wm = 8 (angles)
            # print(abs(wurfs1[i] - wurfs2[j]))
            # dist = -math.sqrt(1 - (abs(wurfs1[i] - wurfs2[j]))**2/4.01)

            if prev_i_j is None:
                return dist
            else:
                prev_i, prev_j = prev_i_j
                return dist \
                    * math.sqrt(i - prev_i + 1) \
                    * math.sqrt(j - prev_j + 1)

        path_dist[0][0] = w_dist(0, 0, None)
        for j in range(1, len(wurfs2)):
            path_dist[0][j] = float("Inf")
        for i in range(1, len(wurfs1)):
            path_dist[i][0] = path_dist[i-1][0] + w_dist(i, 0, (i-1, 0))
            prevs[i][0] = (i-1, 0)

        for i in range(1, len(wurfs1)):
            for j in range(1, len(wurfs2)):
                prev_i = path_dist[i-1][j]
                dist_i = prev_i + w_dist(i, j, (i-1, j))

                prev_j = path_dist[i][j-1] - w_dist(i, j-1, prevs[i][j-1])
                dist_j = prev_j + w_dist(i, j, prevs[i][j-1])

                if dist_i <= prev_j:
                    path_dist[i][j] = dist_i
                    prevs[i][j] = (i-1, j)
                else:
                    path_dist[i][j] = dist_j
                    prevs[i][j] = prevs[i][j-1]

        return path_dist[-1][-1]


    # elif args.diff_method in ["306"]:
    elif args.diff_method in ["5", "COVC"]:
        # ~covariance
        cors = []
        for shift in range(len(wurfs2)):
            shifted_wurfs2 = wurfs2[shift:] + wurfs2[:shift]
            cors.append(abs(calc_complex_correlation(wurfs1, shifted_wurfs2)))

        ac1 = abs(calc_complex_correlation(wurfs1, wurfs1))
        ac2 = abs(calc_complex_correlation(wurfs2, wurfs2))

        m_value = max(cors) / math.sqrt(ac1*ac2)
        return 1 - m_value

    # elif args.diff_method in ["307"]:
    elif args.diff_method in ["4", "COV"]:
        # ~covariance
        cors = []
        cors.append(abs(calc_complex_correlation(wurfs1, wurfs2)))

        ac1 = abs(calc_complex_correlation(wurfs1, wurfs1))
        ac2 = abs(calc_complex_correlation(wurfs2, wurfs2))

        m_value = max(cors) / math.sqrt(ac1*ac2)
        return 1 - m_value

    else:
        raise IOError("unexpected method")


def calc_metrics(diff_values, args):
    err = 0

    n1 = len(diff_values)
    n2 = len(diff_values[0])

    if args.use_metrics == 1:
        for i in range(min(n1, n2)):
            j = i
            val = diff_values[i][j]

            col_other = [diff_values[n_i][j] for n_i in range(n1) if n_i != i]
            row_other = [diff_values[i][n_j] for n_j in range(n2) if n_j != j]

            # print(col_other, row_other)
            # print(err, val)

            err += (val/min(col_other))**2
            err += (val/min(row_other))**2


            if min(col_other) <= val:
                err += 100
            if min(row_other) <= val:
                err += 100

        return err / min(n1,n2)

    elif args.use_metrics == 2:
        for i in range(min(n1, n2)):
            j = i
            val = diff_values[i][j]

            col_other = [diff_values[n_i][j] for n_i in range(n1) if n_i != i]
            row_other = [diff_values[i][n_j] for n_j in range(n2) if n_j != j]

            # print(col_other, row_other)
            # print(err, val)

            err += (val/min(col_other))**4
            err += (val/min(row_other))**4

        return err / min(n1,n2)

    if args.use_metrics == 3:
        for i in range(min(n1, n2)):
            j = i
            val = diff_values[i][j]

            col_other = [diff_values[n_i][j] for n_i in range(n1) if n_i != i]
            row_other = [diff_values[i][n_j] for n_j in range(n2) if n_j != j]

            # print(col_other, row_other)
            # print(err, val)

            # err += (val/min(col_other))**2
            # err += (val/min(row_other))**2


            if min(col_other) <= val:
                err += 1.
            if min(row_other) <= val:
                err += 1.

        return err / (2*min(n1,n2))

    if args.use_metrics == 4:
        for i in range(min(n1, n2)):
            j = i
            val = diff_values[i][j]

            col_other = [diff_values[n_i][j] for n_i in range(n1) if n_i != i]
            row_other = [diff_values[i][n_j] for n_j in range(n2) if n_j != j]

            # print(col_other, row_other)
            # print(err, val)

            err += (min(col_other)/(val + min(col_other)))
            err += (min(row_other)/(val + min(row_other)))



        return err / (2*min(n1,n2))


    else:
        raise IOError("unexpected method")

def main():
    args = parse_args()
    drawer = None
    if not args.no_image:
        tk, drawer = init_tk_drawer()

    values_for_files_1 = calc_and_draw_values(drawer, args.files, args)
    if args.files_other:
        values_for_files_2 = calc_and_draw_values(drawer, args.files_other, args, fill='blue')

        n1 = len(values_for_files_1)
        n2 = len(values_for_files_2)


        diff_values = [[0]*n2 for i in range(n1)]
        for i in range(n1):
            for j in range(n2):
                diff_values[i][j] = calc_diff(values_for_files_1[i], values_for_files_2[j], args)

        if args.use_metrics == 0:
            print("\t".join([""] + [str(i+1) for i in range(n2)]))
            for i in range(n1):
                s = "{i}\t".format(i=i+1)

                for j in range(n2):
                    s += str(diff_values[i][j]) + "\t"

                print(s)
        else:
            print(calc_metrics(diff_values, args))




        # for p in points[::100]:
        #     drawer.draw_circle(p)




    def zoom( event):
        print("Hello windows/macos! Not-tested scaling.")
        drawer.scale(1.1 ** event.delta, event.x, event.y)

    def zoom_in( event):
        drawer.scale(1.1, event.x, event.y)

    def zoom_out( event):
        drawer.scale(1.1 ** (-1), event.x, event.y)


    if not args.no_image:
        tk.bind("<MouseWheel>", zoom)
        tk.bind("<Button-4>", zoom_in)
        tk.bind("<Button-5>", zoom_out)
        tk.mainloop()


if __name__ == "__main__":
    main()


