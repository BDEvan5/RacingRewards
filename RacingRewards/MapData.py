
import numpy as np
from PIL import Image
import csv 
import yaml
from numba import njit

class MapData:
    def __init__(self, map_name):
        self.map_name = map_name

        self.cline = None
        self.nvecs = None
        self.widths = None
        self.N = None

        self.resolution = None
        self.origin = None
        # self.stheta = yaml_file['start_pose'][2]
        # self.stheta = -np.pi/2
        self.map_img_name = None

        self.map_img = None
        self.height = None
        self.width = None

        self.read_yaml_file()
        self.load_track_pts()
        self.load_map()


    def load_track_pts(self):
        track = []
        filename = 'maps/' + self.map_name + "_std.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename} in env_map")

        self.N = len(track)
        self.cline = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.widths = track[:, 4:6]
        
    def read_yaml_file(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

        yaml_file = dict(documents.items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        # self.stheta = yaml_file['start_pose'][2]
        self.stheta = -np.pi/2
        self.map_img_name = yaml_file['image']

    def load_map(self):
        map_img_name = 'maps/' + self.map_img_name

        try:
            self.map_img = np.array(Image.open(map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        except Exception as e:
            print(f"MapPath: {map_img_name}")
            print(f"Exception in reading: {e}")
            raise ImportError(f"Cannot read map")
        try:
            self.map_img = self.map_img[:, :, 0]
        except:
            pass

        self.height = self.map_img.shape[1]
        self.width = self.map_img.shape[0]


    def _expand_wpts(self):
        n = 5 # number of pts per orig pt 
        dz = 1 / n
        o_line = self.cline
        # o_vs = self.vs
        new_line = []
        # new_vs = []
        for i in range(len(o_line)-1):
            dd = sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                # dv = o_vs[i+1] - o_vs[i]
                # new_vs.append(o_vs[i] + dv * j * dz)

        self.cline = np.array(new_line)
        self.N = len(self.cline)
        # self.vs = np.array(new_vs)


@njit(fastmath=True, cache=True)
def add_locations(x1, x2, dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret

@njit(fastmath=True, cache=True)
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret


