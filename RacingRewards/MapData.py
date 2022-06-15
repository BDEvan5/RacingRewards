
import numpy as np
from PIL import Image
import csv 
import yaml

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



