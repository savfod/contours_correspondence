import math
import sys
if sys.version_info >= (3, 0):
    import tkinter
else:
    import Tkinter as tkinter

WIDTH = 1000
HEIGHT = 700

MIN_POINT_DISTANCE = 0

def plane_transformation(x,y):
    # x = WIDTH*x/2 + WIDTH/2
    x = WIDTH*x/1.1 + WIDTH/20
    y = -HEIGHT*y/1.1 + HEIGHT/1.05
    return (x,y)


CYLINDER_HEIGHT_FRAC = 3
CYL_COEFF = 3

def cylinder_transformation(x,y):
    # angle = 2*math.atan(CYL_COEFF*x)
    angle = (math.exp(CYL_COEFF*x)/(1 + math.exp(CYL_COEFF*x)) - 0.5) * 2*math.pi
    return plane_transformation(math.sin(angle), (y - math.cos(angle)/CYLINDER_HEIGHT_FRAC)/2)

def cylinder_on_back_side(x,y):
    # angle = 2*math.atan(CYL_COEFF*x)
    angle = (math.exp(CYL_COEFF*x)/(1 + math.exp(CYL_COEFF*x)) - 0.5) * 2*math.pi
    return abs(angle) > math.pi/2

def calc_cylynder_y0_cross_section_sizes():
    xs = []
    ys = []
    power = 3
    for i in range(-10**power, 10**power, 10**(power - 2)):
        x, y = cylinder_transformation(i/100., 0)
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


PLANE = 1
CYLINDER = 2


class Drawer:

    def __init__(self):
        def add_canvas(plane_type, labels=["", ""]):
            index = len(self.canvases.keys()) + 1

            canvas = tkinter.Canvas(self.tk, width=WIDTH, height=HEIGHT)
            canvas.configure(background='white')
            canvas.pack()

            self.canvases[index] = canvas
            self.circles_centers[index] = []
            self.canvas_types[index] = plane_type

            if plane_type == CYLINDER:
                transform_function = cylinder_transformation
            else:
                transform_function = plane_transformation
            self.canvas_transformation_functions[index] = transform_function

            if plane_type == CYLINDER:
                x_min, y_min, x_max, y_max = calc_cylynder_y0_cross_section_sizes()
                canvas.create_oval(x_min, y_min + HEIGHT/4, x_max, y_max + HEIGHT/4, fill="yellow")
                y_mid = (y_min + y_max) / 2
                canvas.create_rectangle(x_min, y_mid - HEIGHT/4, x_max, y_mid + HEIGHT/4, fill="yellow")
                canvas.create_oval(x_min, y_min - HEIGHT/4, x_max, y_max - HEIGHT/4, fill="orange")

            self.draw_arrows(img_index=index, labels=labels)

        self.tk = tkinter.Tk()
        self.canvases = {}
        self.circles_centers = {}
        self.canvas_transformation_functions = {}
        self.canvas_types = {}

        add_canvas(plane_type=PLANE, labels=["x", "y"])
        # add_canvas(plane_type=PLANE, labels=["p.x", "t.x"])
        # add_canvas(plane_type=CYLINDER, labels=["p.x", "t.x"])

        self.tmp_objects = {}

    def add_tmp(self, img_index, obj):
        if img_index not in self.tmp_objects:
            self.tmp_objects[img_index] = []
        self.tmp_objects[img_index].append(obj)

    def draw_arrows(self, labels=["x","y"], img_index=1):
        DRAW_INDENT = 15
        sxsy = self.canvas_transformation_functions[img_index]
        canvas = self.canvases[img_index]

        for i in range(1,100):
            canvas.create_line(sxsy((i-1)/100.,0), sxsy(i/100.,0))
        canvas.create_line(sxsy(99./100,0), sxsy(1,0), arrow=tkinter.LAST)
        canvas.create_text(sxsy(1,0)[0]-DRAW_INDENT, sxsy(1,0)[1]+DRAW_INDENT, text=labels[0], font=("Arial", "15"))

        for i in range(1,100):
            canvas.create_line(sxsy(0,(i-1)/100.), sxsy(0, i/100.))
        canvas.create_line(sxsy(0,0), sxsy(0,1), arrow=tkinter.LAST)
        canvas.create_text(sxsy(0,1)[0]-DRAW_INDENT, sxsy(0,1)[1]+DRAW_INDENT, text=labels[1], font=("Arial", "15"))

    def draw_line(self, p1, p2, fill="black", img_index=1, tmp_object=False, width=2, dash=None):
        sxsy = self.canvas_transformation_functions[img_index]
        canvas = self.canvases[img_index]
        if img_index == 3 and cylinder_on_back_side(p2.x, p2.y):
            if not 0.2 <= p1.y <= 0.4:
                return

            if abs(p1-p2) > 0.05:
                dash = [4, 8]
            else:
                if (p1.x + p1.y) % 0.1 < 0.07:
                    return
        obj = canvas.create_line(sxsy(p1.x, p1.y), sxsy(p2.x, p2.y), fill=fill, width=width, dash=dash)
        if tmp_object:
            self.add_tmp(img_index, obj)
        return obj

    def draw_circle(self, center, fill="blue", r=2, img_index=1, tmp_object=False, label=None, c_x=4, c_y=8):
        if img_index not in self.canvases:
            return

        if (not tmp_object) and any(abs(center - x) < MIN_POINT_DISTANCE for x in self.circles_centers[img_index]):
            return

        sxsy = self.canvas_transformation_functions[img_index]
        canvas = self.canvases[img_index]
        cx, cy = sxsy(center.x, center.y)

        if img_index == 3 and cylinder_on_back_side(center.x, center.y):
            fill = "dark " + fill

        obj = canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=fill)
        if tmp_object:
            self.add_tmp(img_index, obj)

        if not tmp_object:
            self.circles_centers[img_index].append(center)

        if label:
            obj2 = canvas.create_text(cx-c_x*r, cy+c_y*r, fill=fill, text=label, font=("Arial", "15"))
            if tmp_object:
                self.add_tmp(img_index, obj2)

        return obj

    def remove_tmp(self, img_index=-1):
        if img_index == -1: #all
            for k in self.tmp_objects.keys():
                self.remove_tmp(k)
        else:
            canvas = self.canvases[img_index]
            canvas.delete(*self.tmp_objects[img_index])
            self.tmp_objects[img_index] = []

    def remove_all(self):
        return self.canvas.delete(*self.canvas.find_all())

    def draw_curve(self, curve, fill="pink", img_index=1):
        points = curve.get_all_points()
        for p in points:
            #self.draw_circle(p, fill=fill, img_index=img_index)
            pass
        for (p1,p2) in zip(points, points[1:]):
            self.draw_line(p1, p2, fill=fill, img_index=img_index)

    def scale(self, k, x_offset=WIDTH//2, y_offset=HEIGHT//2):
        for c in self.canvases.values():
            c.scale("all", x_offset, y_offset, k, k)

    def remove(self, objs):
        for k in self.canvases.values():
            k.delete(*objs)
