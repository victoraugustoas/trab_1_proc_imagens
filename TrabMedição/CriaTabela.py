import csv
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from datetime import datetime

fileCabo = "Medição - CabCasa.csv"
file2Wifi = "Medição - Wifi.csv"


def extrai_dados(file):
    """
    Função para retornar os dados presentes no csv contendo os dados da coleta
    :param file: arquivo csv contendo os campos: Server, Data, Hora, Período, tipo_conexão, ping, download, upload
    :return: 5 listas contendo os dados referentes aos 3 peridos (manha, tarde, noite, download, upload)
    """
    with open(file, encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        return list(csv_reader)


tuplaCabo = extrai_dados(fileCabo)
tuplaWF = extrai_dados(file2Wifi)

#   0    1       2           3          4
# (data, hora, int(ping), float(dl), float(up))
# organizando os dados provenientes das funções
manha_cabo = list(filter(lambda x: x["Período"] == "manhã", tuplaCabo))
tarde_cabo = list(filter(lambda x: x["Período"] == "tarde", tuplaCabo))
noite_cabo = list(filter(lambda x: x["Período"] == "noite", tuplaCabo))
dl_cabo = list(map(lambda x: float(x["download"]), tuplaCabo))
up_cabo = list(map(lambda x: float(x["upload"]), tuplaCabo))


manhawf = list(filter(lambda x: x["Período"] == "manhã", tuplaWF))
tardewf = list(filter(lambda x: x["Período"] == "tarde", tuplaWF))
noitewf = list(filter(lambda x: x["Período"] == "noite", tuplaWF))
dl_wf = list(map(lambda x: float(x["download"]), tuplaWF))
up_wf = list(map(lambda x: float(x["upload"]), tuplaWF))


# plotando grafico cabo cronológico do periodo analisado Cabo vs Wifi
# download todas as medições
aux1 = np.array(dl_cabo)
aux2 = np.array(dl_wf)
plt.plot(aux1, "b", label="cabeada")
wf = plt.plot(aux2, "g", label="wifi")
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Download")
plt.legend(loc="best")
plt.show()

# upload todas as medições
aux1 = np.array(up_cabo)
aux2 = np.array(up_wf)
plt.plot(aux1, label="cabeada")
wf = plt.plot(aux2, label="wifi")
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Upload")
plt.legend(loc="best")
plt.show()


# medições de DOWNLOAD por periodo CABO
manha_dl = np.array(list(map(lambda x: float(x["download"]), manha_cabo)))
tarde_dl = np.array(list(map(lambda x: float(x["download"]), tarde_cabo)))
noite_dl = np.array(list(map(lambda x: float(x["download"]), noite_cabo)))

plt.subplot(311)
plt.plot(manha_dl)
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Download Cabo Manhã/Tarde/Noite")
plt.subplot(312)
plt.plot(tarde_dl, "y")
plt.ylabel("velocidade (Mbps)")
plt.subplot(313)
plt.plot(noite_dl, "r")
plt.ylabel("velocidade (Mbps)")
plt.show()

# medições de UPLOAD por periodo CABO
manha_up = np.array(list(map(lambda x: float(x["upload"]), manha_cabo)))
tarde_up = np.array(list(map(lambda x: float(x["upload"]), tarde_cabo)))
noite_up = np.array(list(map(lambda x: float(x["upload"]), noite_cabo)))
plt.subplot(311)
plt.plot(manha_up)
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Upload Cabo Manhã/Tarde/Noite")
plt.subplot(312)
plt.plot(tarde_up, "y")
plt.ylabel("velocidade (Mbps)")
plt.subplot(313)
plt.plot(noite_up, "r")
plt.ylabel("velocidade (Mbps)")
plt.show()
############################################################################
############################################################################
############################################################################


# medições de DOWNLOAD por periodo WIFI
manha_dl = np.array(list(map(lambda x: float(x["download"]), manhawf)))
tarde_dl = np.array(list(map(lambda x: float(x["download"]), tardewf)))
noite_dl = np.array(list(map(lambda x: float(x["download"]), noitewf)))

plt.subplot(311)
plt.plot(manha_dl)
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Download Wifi Manhã/Tarde/Noite")
plt.subplot(312)
plt.plot(tarde_dl, "y")
plt.ylabel("velocidade (Mbps)")
plt.subplot(313)
plt.plot(noite_dl, "r")
plt.ylabel("velocidade (Mbps)")
plt.show()

# medições de UPLOAD por periodo WIFI
manha_up = np.array(list(map(lambda x: float(x["upload"]), manhawf)))
tarde_up = np.array(list(map(lambda x: float(x["upload"]), tardewf)))
noite_up = np.array(list(map(lambda x: float(x["upload"]), noitewf)))

plt.subplot(311)
plt.plot(manha_up)
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Upload Wifi Manhã/Tarde/Noite")
plt.subplot(312)
plt.plot(tarde_up, "y")
plt.ylabel("velocidade (Mbps)")
plt.subplot(313)
plt.plot(noite_up, "r")
plt.ylabel("velocidade (Mbps)")
plt.show()

#############################Grafico ping######################################
# plotando grafico do ping
ping_cabo = np.array(list(map(lambda x: float(x["ping"]), tuplaCabo)))
ping_wifi = np.array(list(map(lambda x: float(x["ping"]), tuplaWF)))

plt.subplot(211)
plt.plot(ping_cabo, "pink")
plt.ylabel("Ping (ms)")
plt.title("Latência Cabo/Wifi")
plt.subplot(212)
plt.plot(ping_wifi, "orange")
plt.xlabel("Nº Medição")
plt.ylabel("Ping (ms)")
plt.show()

#################################Gráfico dl/up dia de semana vs fim de semana######################
#   0    1       2           3          4
# (data, hora, int(ping), float(dl), float(up))
# organizando os dados provenientes das funções
# listas com as informações da coleta nos dias de semana e nos fins de semana cabo

semana = list(
    filter(
        lambda x: True
        if x["Data"] != "19/10/2019" and x["Data"] != "20/10/2019"
        else False,
        tuplaCabo,
    )
)

fds = list(
    filter(
        lambda x: True
        if x["Data"] == "19/10/2019" or x["Data"] == "20/10/2019"
        else False,
        tuplaCabo,
    )
)
dl_semana_cabo = list(map(lambda x: float(x["download"]), semana))
up_semana_cabo = list(map(lambda x: float(x["upload"]), semana))
dl_fds_cabo = list(map(lambda x: float(x["download"]), fds))
up_fds_cabo = list(map(lambda x: float(x["upload"]), fds))


semana = list(
    filter(
        lambda x: True
        if x["Data"] != "19/10/2019" and x["Data"] != "20/10/2019"
        else False,
        tuplaWF,
    )
)

fds = list(
    filter(
        lambda x: True
        if x["Data"] == "19/10/2019" or x["Data"] == "20/10/2019"
        else False,
        tuplaWF,
    )
)

dl_fds_wifi = list(map(lambda x: float(x["download"]), fds))
up_fds_wifi = list(map(lambda x: float(x["upload"]), fds))
dl_semana_wifi = list(map(lambda x: float(x["download"]), semana))
up_semana_wifi = list(map(lambda x: float(x["upload"]), semana))

# dl vs dl semana/fds
plt.subplot(211)
plt.plot(dl_semana_cabo, "brown")
plt.ylabel("Velocidade (Mbps)")
plt.title("Download Cabo Semana/FDS")
plt.subplot(212)
plt.plot(dl_fds_cabo, "cyan")
plt.xlabel("Nº Medição")
plt.ylabel("Velocidade (Mbps)")
plt.show()

plt.subplot(211)
plt.plot(up_semana_cabo, "brown")
plt.ylabel("Velocidade (Mbps)")
plt.title("Upload Cabo Semana/FDS")
plt.subplot(212)
plt.plot(up_fds_cabo, "cyan")
plt.xlabel("Nº Medição")
plt.ylabel("Velocidade (Mbps)")
plt.show()

##########################################
##########################################
##########################################

# dl vs dl semana/fds
plt.subplot(211)
plt.plot(dl_semana_wifi, "green")
plt.ylabel("Velocidade (Mbps)")
plt.title("Download Wifi Semana/FDS")
plt.subplot(212)
plt.plot(dl_fds_wifi, "black")
plt.xlabel("Nº Medição")
plt.ylabel("Velocidade (Mbps)")
plt.show()

plt.subplot(211)
plt.plot(up_semana_wifi, "green")
plt.ylabel("Velocidade (Mbps)")
plt.title("Upload Wifi Semana/FDS")
plt.subplot(212)
plt.plot(up_fds_wifi, "black")
plt.xlabel("Nº Medição")
plt.ylabel("Velocidade (Mbps)")
plt.show()
