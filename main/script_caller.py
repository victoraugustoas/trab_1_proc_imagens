from datetime import datetime

import proc_video as cv4
import utills
from pprint import pprint

PATH_VIDEO = "../videos"

limiar = [30, 40, 50, 60, 70, 80, 90]
tam_img = [20, 40]
grid = [5, 10]
algs = ["bic_dlog", "bic", "hist"]

now = datetime.now()
dt_string = now.strftime("%H:%M:%S")
print("inicio =", dt_string)


def to_arq(
    video,
    csv,
    shots,
    percent_limiar,
    tam_img_percent,
    str_alg,
    type_function,
    tam_grid_str,
):
    aux = cv4.compare_times(video, csv, shots)
    # escrevendo a precisao do trem num arquivo
    with open(
        "./data/filevd_"
        + str(i)
        + "_"
        + str(percent_limiar)
        + "_"
        + str(tam_img_percent)
        + "_"
        + str_alg
        + ".txt",
        "w",
        encoding="utf-8",
    ) as saida:

        saida.write("video: " + str(i) + "\n")
        saida.write("percent limiar: " + str(percent_limiar) + "\n")
        saida.write("tam_img:" + str(tam_img_percent) + "\n")
        saida.write("algoritmo:" + str_alg + "\n")
        saida.write("type function:" + type_function + "\n")
        saida.write("tam grid:" + tam_grid_str + "\n")

        for linha in shots:
            saida.write(str(linha))
            saida.write("\n")
        saida.write("Accuracy: " + str(aux) + "\n")


for i in range(5, 6):
    print("processando o video ", i)
    i = str(i)
    str_alg = ""
    video_path = "../videos/video" + i + ".mp4"
    csv_path = "../videos/cortes_video" + i + ".csv"

    for percent_limiar in limiar:
        for tam_img_percent in tam_img:
            for alg in algs:
                video = cv4.open_video(video_path)
                csv = utills.csv_to_dict(csv_path)

                frames = cv4.get_frames(
                    video,
                    cv4.status_video(video).get("fps"),
                    resize={"percent": tam_img_percent},
                )

                if alg == "bic":
                    str_alg = "bic"
                    alg = cv4.similarity_bic
                elif alg == "bic_dlog":
                    str_alg = alg
                    alg = cv4.similarity_bic_dlog
                else:
                    str_alg = "hist"
                    alg = cv4.similarity_hist
                # print(len(frames))
                for type_function in ["grid", "normal"]:

                    print("processando video:")
                    limit = percent_limiar / 100
                    tam_grid_str = None

                    if type_function == "grid":
                        for tam_grid in grid:
                            pprint(
                                {
                                    "video": i,
                                    "percent_limiar": percent_limiar,
                                    "tam_img_percent": tam_img_percent,
                                    "alg": str_alg,
                                    "type_function": type_function,
                                    "grid": tam_grid,
                                }
                            )
                            tam_grid_str = str(tam_grid)
                            shots = cv4.shot_boundary_detection_grid(
                                video, frames, alg, limit=limit, nparts=tam_grid
                            )

                            to_arq(
                                video,
                                csv,
                                shots,
                                percent_limiar,
                                tam_img_percent,
                                str_alg,
                                type_function,
                                tam_grid_str,
                            )
                    else:
                        pprint(
                            {
                                "video": i,
                                "percent_limiar": percent_limiar,
                                "tam_img_percent": tam_img_percent,
                                "alg": str_alg,
                                "type_function": type_function,
                            }
                        )
                        shots = cv4.shot_boundary_detection(
                            video, frames, alg, limit=limit
                        )
                        to_arq(
                            video,
                            csv,
                            shots,
                            percent_limiar,
                            tam_img_percent,
                            str_alg,
                            type_function,
                            tam_grid_str,
                        )


now = datetime.now()
dt_string = now.strftime("%H:%M:%S")
print("fim =", dt_string)
