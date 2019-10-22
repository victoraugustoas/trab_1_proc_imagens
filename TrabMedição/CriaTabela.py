import csv
import matplotlib.pyplot as plt
import numpy as np

fileCabo = "Medição - CabCasa.csv"
file2Wifi = "Medição - Wifi.csv"


def extrai_dados(file):
    """
    Função para retornar os dados presentes no csv contendo os dados da coleta
    :param file: arquivo csv contendo os campos: Server, Data, Hora, Período, tipo_conexão, ping, download, upload
    :return: 5 listas contendo os dados referentes aos 3 peridos (manha, tarde, noite, download, upload)
    """
    with open(file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        l_manha = []
        l_tarde = []
        l_noite = []
        l_dl = []
        l_up = []
        for row in csv_reader:
            if line_count == 0:
                cabeçalho = row
                line_count += 1
            else:
                data = row[1]
                hora = row[2]
                periodo = row[3]
                ping = row[5]
                dl = row[6]
                up = row[7]

                tupla = (data, hora, int(ping), float(dl), float(up))
                l_dl.append(float(dl))
                l_up.append(float(up))
                if periodo == "manhã":
                    l_manha.append(tupla)
                elif periodo == "tarde":
                    l_tarde.append(tupla)
                else:
                   l_noite.append(tupla)
                line_count += 1
        return l_manha, l_tarde, l_noite, l_dl, l_up

tuplaCabo = extrai_dados(fileCabo)
tuplaWF = extrai_dados(file2Wifi)

#   0    1       2           3          4
#(data, hora, int(ping), float(dl), float(up))
#organizando os dados provenientes das funções
manha_cabo = tuplaCabo[0]
tarde_cabo = tuplaCabo[1]
noite_cabo = tuplaCabo[2]
dl_cabo = tuplaCabo[3]
up_cabo = tuplaCabo[4]

manhawf = tuplaWF[0]
tardewf = tuplaWF[1]
noitewf = tuplaWF[2]
dl_wf = tuplaWF[3]
up_wf = tuplaWF[4]

print(len(manha_cabo), len(tarde_cabo), len(noite_cabo), len(manha_cabo) + len(tarde_cabo) + len(noite_cabo))
print(len(dl_cabo), len(up_cabo))
print(len(manhawf), len(tardewf), len(noitewf), len(manhawf) + len(tardewf) + len(noitewf))
print(len(dl_wf), len(up_wf))

#plotando grafico cabo cronológico do periodo analisado Cabo vs Wifi
#download todas as medições
aux1 = np.array(dl_cabo)
aux2 = np.array(dl_wf)
plt.plot(aux1, "b", label = "cabeada")
wf = plt.plot(aux2, "g", label = "wifi")
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Download")
plt.legend(loc="best")
plt.show()

#upload todas as medições
aux1 = np.array(up_cabo)
aux2 = np.array(up_wf)
plt.plot(aux1, label = "cabeada")
wf = plt.plot(aux2, label = "wifi")
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Upload")
plt.legend(loc="best")
plt.show()


manha_dl = []
manha_up = []
tarde_dl = []
tarde_up = []
noite_dl = []
noite_up = []
for elemento in manha_cabo:
    manha_dl.append(elemento[3])
    manha_up.append(elemento[4])
for elemento in tarde_cabo:
    tarde_dl.append(elemento[3])
    tarde_up.append(elemento[4])
for elemento in noite_cabo:
    noite_dl.append(elemento[3])
    noite_up.append(elemento[4])

#medições de DOWNLOAD por periodo CABO
manha_dl = np.array(manha_dl)
tarde_dl = np.array(tarde_dl)
noite_dl = np.array(noite_dl)

plt.subplot(311)
plt.plot(manha_dl)
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Download Manhã/Tarde/Noite")
plt.subplot(312)
plt.plot(tarde_dl, "y")
plt.ylabel("velocidade (Mbps)")
plt.subplot(313)
plt.plot(noite_dl, "r")
plt.ylabel("velocidade (Mbps)")
plt.show()

#medições de UPLOAD por periodo CABO
manha_up = np.array(manha_up)
tarde_up = np.array(tarde_up)
noite_up = np.array(noite_up)
plt.subplot(311)
plt.plot(manha_up)
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Upload Manhã/Tarde/Noite")
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

manha_dl = []
manha_up = []
tarde_dl = []
tarde_up = []
noite_dl = []
noite_up = []
for elemento in manhawf:
    manha_dl.append(elemento[3])
    manha_up.append(elemento[4])
for elemento in tardewf:
    tarde_dl.append(elemento[3])
    tarde_up.append(elemento[4])
for elemento in noitewf:
    noite_dl.append(elemento[3])
    noite_up.append(elemento[4])

#medições de DOWNLOAD por periodo WIFI
manha_dl = np.array(manha_dl)
tarde_dl = np.array(tarde_dl)
noite_dl = np.array(noite_dl)

plt.subplot(311)
plt.plot(manha_dl)
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Download Manhã/Tarde/Noite")
plt.subplot(312)
plt.plot(tarde_dl, "y")
plt.ylabel("velocidade (Mbps)")
plt.subplot(313)
plt.plot(noite_dl, "r")
plt.ylabel("velocidade (Mbps)")
plt.show()

#medições de UPLOAD por periodo WIFI
manha_up = np.array(manha_up)
tarde_up = np.array(tarde_up)
noite_up = np.array(noite_up)

plt.subplot(311)
plt.plot(manha_up)
plt.xlabel("Nº Medição")
plt.ylabel("velocidade (Mbps)")
plt.title("Velocidade Upload Manhã/Tarde/Noite")
plt.subplot(312)
plt.plot(tarde_up, "y")
plt.ylabel("velocidade (Mbps)")
plt.subplot(313)
plt.plot(noite_up, "r")
plt.ylabel("velocidade (Mbps)")
plt.show()