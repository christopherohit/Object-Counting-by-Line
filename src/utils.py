# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon

def reconfig_logfile(path_to_log: str = ''):
    total_log = []
    with open(path_to_log,'r') as f:
        for line in f:
            total_log.append(line.strip())
