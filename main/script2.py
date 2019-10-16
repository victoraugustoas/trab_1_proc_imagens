import os

PATH_VIDEO = "../videos"

limiar = [30, 40, 50, 60, 70, 80, 90]
tam_img = [15, 40]
algs = ["bic", "hist"]

for i in range(1, 11):
    i = str(i)
    video_path = os.path.join(PATH_VIDEO, "video_" + i + ".mp4")
    csv_path = os.path.join(PATH_VIDEO, "cortes_video" + i + ".csv")

    for percent_limiar in limiar:
        for tam_img_percent in tam_img:
            for alg in algs:
                name_session = (
                    "video_"
                    + i
                    + "_"
                    + str(tam_img_percent)
                    + "_"
                    + str(percent_limiar)
                )
                command = "python3 script1.py %d %d %s %s %s > ./data/%s.txt" % (
                    tam_img_percent,
                    percent_limiar,
                    video_path,
                    csv_path,
                    alg,
                    name_session,
                )

                # abre o comando em um terminal virtual para execução com onome igual video_X_tam_limiar
                cmd = "tmux new -ds %s '%s'" % (name_session, command)
                print(cmd)
                os.system(cmd)

                break
