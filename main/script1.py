import proc_img as cv3
import cv2
import proc_video as cv4
from pprint import pprint
import utills
import sys

"""
Parametros devem ser usados nessa ordem

Params:
1 - percent video
2 - limit
3 - path video
4 - path csv
5 - algoritmo
"""
'''
percent_video = int(sys.argv[1])
limit = float(sys.argv[2])
path_video = sys.argv[3]
path_csv = sys.argv[4]
alg = sys.argv[5'''

percent_video = 15
limit = 0.90
path_video = "../videos/video1.mp4"
path_csv = "../videos/cortes_video1.csv"
alg = "bic"

if alg == "bic":
    alg = cv4.similarity_bic
else:
    alg = cv4.similarity_hist

PATH_IMG = "../imagens/aviao.jpg"
PATH_VIDEO = "../videos/video5.mp4"
PATH_CSV = "../videos/cortes_video5.csv"
PATH_DATA = "./data"

video = cv4.open_video(path_video)
csv = utills.csv_to_dict(path_csv)

frames = cv4.get_frames(
    video, cv4.status_video(video).get("fps"), resize={"percent": percent_video}
)

print(len(frames))
shots = cv4.shot_boundary_detection(video, frames, alg, limit=limit)

print(cv4.compare_times(video, csv, shots))
