import proc_video as cv4
import utills
from datetime import datetime
# datetime object containing current date and time

PATH_VIDEO = "../videos"

limiar = 90
tam_img = 20
algs = "hist"

now = datetime.now()
dt_string = now.strftime("%H:%M:%S")
print("inicio =", dt_string)	

for i in range(1, 5):
    print("processando o video ", i)
    i = str(i)
    str_alg = ""
    video_path = ("../videos/video" + i + ".mp4")
    csv_path = ( "../videos/cortes_video" + i + ".csv")

    video = cv4.open_video(video_path)
    csv = utills.csv_to_dict(csv_path)

    frames = cv4.get_frames(
        video, cv4.status_video(video).get("fps"), resize={"percent": tam_img_percent}
        )

    alg = cv4.similarity_hist
    #print(len(frames))
    print("processando video:", i, percent_limiar, tam_img_percent, alg)
    shots = cv4.shot_boundary_detection(video, frames, alg, 0.9)
    #print(shots)
    #shots = cv4.shot_boundary_detection_grid(video, frames, alg, limit)
    aux = cv4.compare_times(video, csv, shots)
    #escrevendo a precisao do trem num arquivo
    with open("expBic/data/filevd_" + str(i) + ".txt", "w", encoding="utf-8") as saida:  
        saida.write("video: " + str(i) + "\n")
        saida.write("percent limiar: " + str(limiar) + "\n" )
        saida.write("tam_img:" + str(tam_img) + "\n" )
        saida.write("algoritmo:" + "hist" + "\n")
        
        for linha in shots:
            saida.write(str(linha))
            saida.write("\n")
        saida.write("Accuracy: " + str(aux) + "\n")

now = datetime.now()
dt_string = now.strftime("%H:%M:%S")
print("fim =", dt_string)
                