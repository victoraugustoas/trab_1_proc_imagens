import platform
from datetime import datetime as dt
from datetime import timedelta
from pprint import pprint
import os

import cv2
import numba as nb
import numpy as np

# verifica qual sistema operacional está rodando
try:
    if platform.linux_distribution():
        import proc_img as cv3
except:
    import trab_1_proc_imagens.main.proc_img as cv3


def open_video(path):
    """
        Abre a imagem e retorna o obj do video
    """
    return cv2.VideoCapture(path)


def display_video(video, percent=50, dim=None):
    """
        Reproduz o video

        Para parar a reprodução, aperte Q

        percent: valor inteiro entre 0 e 100
        dim: tupla com valores inteiros de largura e altura respectivamente
    """

    # ocorreu um erro ao abrir o arquivo
    if not video.isOpened():
        raise Exception("Video não pode ser aberto")

    print("Pressione Q para sair da exibição")
    while video.isOpened():
        next_frame, frame = video.read()

        if next_frame == True:
            # mostra o frame atual
            cv2.imshow("Frame", cv3.resize_img(frame, percent=percent, dim=dim))

            # Pressione Q para parar a reprodução
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break

    # fecha todas as janelas
    cv2.destroyAllWindows()


def status_video(video):
    """
        Retorna um objeto com as seguintes propriedades:
        width, height e fps
    """
    return {
        "width": video.get(cv2.CAP_PROP_FRAME_WIDTH),  # float
        "height": video.get(cv2.CAP_PROP_FRAME_HEIGHT),  # float
        "fps": video.get(cv2.CAP_PROP_FPS),  # float,
    }


def get_time_frame(video, frame_id):
    """
        Retorna o tempo em hora:minuto:segundo.milisegundo

        frame_id: id do frame começando em 0

        retorna uma string
    """

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ms = video.get(cv2.CAP_PROP_POS_MSEC)

    return str(timedelta(milliseconds=ms))


def get_frame(video, frame_id):
    """
        Dado um id do frame, retorna o frame correspondente
    """
    # salva o frame atual
    frame_atual = video.get(cv2.CAP_PROP_POS_FRAMES)

    # muda para o frame correspondente
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    read_flag, frame = video.read()

    # retorna o frame atual
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_atual)

    return frame


@nb.jit
def get_frames(video, fps, resize={}):
    """
        função para extrair os frames de um video, nessa função especificamente serão retornado os frames em um intervalo,
        ou seja, a cada fps frames 1 será retornado.
        
        :param video: video que deve ser aberto para que seja extraido seus frames
        :param fps: a taxa de quadros do video, para que possa ser implementada a janela deslizante
        :return: uma tupla com uma lista contendo os frames, eles já estão prontos para serem exibidos
        e uma lista com os ids dos frames [(frame_id, frame), ..] (((RETORNA UMA LISTA DE DICTS)))
    """

    read_flag, frame = video.read()
    vid_frames = []
    i = 1

    fps = int(fps)

    while read_flag:
        if i % fps == 0:
            if resize.get("dim", None) != None:
                info = {
                    "frame_id": i - 1,
                    "frame": cv3.resize_img(frame, dim=resize.get("dim")),
                }
            elif resize.get("percent", None) != None:
                info = {
                    "frame_id": i - 1,
                    "frame": cv3.resize_img(frame, percent=resize.get("percent")),
                }
            else:
                info = {"frame_id": i - 1, "frame": frame}

            vid_frames.append(info)

        read_flag, frame = video.read()
        i += 1

    return vid_frames


@nb.jit
def cos_vectors(arr_a, arr_b):
    """
        Calcula o cosseno do angulo de dois vetores

        retorna o cosseno do angulo, valor entre 0 e 1
    """
    produto_interno = np.inner(arr_a, arr_b)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)

    cos = produto_interno / (norm_a * norm_b)

    return cos


@nb.jit
def dlog(arr_a, arr_b):
    """
        Calcula a função dlog para 2 frames

        retorna um valor entre 0 e 9
    """
    q = []
    d = []
    for pos_a in arr_a:
        if pos_a == 0:
            q.append(0)
        elif pos_a > 0 and pos_a <= 1:
            q.append(1)
        else:
            q.append(np.log2(pos_a) + 1)

    for pos_b in arr_b:
        if pos_b == 0:
            d.append(0)
        elif pos_b > 0 and pos_b <= 1:
            d.append(1)
        else:
            d.append(np.log2(pos_b) + 1)

    q = np.array(q)
    d = np.array(d)

    return np.sum(abs(q - d))


@nb.jit
def similarity_hist(frame_a, frame_b):
    """
        Calcula a similaridade entre dois frames utilizando o histograma global e o cosseno do angulo de dois vetores

        retorna a similaridade

        referencia:
        https://mundoeducacao.bol.uol.com.br/matematica/angulo-entre-dois-vetores.htm
    """

    hist_a = cv3.histogram_global(frame_a)
    hist_b = cv3.histogram_global(frame_b)

    lst_a = []
    lst_b = []

    for i in range(0, 3):
        for num in hist_a[i]:
            lst_a.append(num)

    for i in range(0, 3):
        for num in hist_b[i]:
            lst_b.append(num)

    array_a = np.array(lst_a)
    array_b = np.array(lst_b)

    return cos_vectors(array_a, array_b)


def similarity_bic(frame_a, frame_b):
    bic_a = cv3.bic(cv3.quantization_colors(frame_a), 32)
    bic_b = cv3.bic(cv3.quantization_colors(frame_b), 32)

    # pega o vetor de caracteristicas do bic se for um dict
    if isinstance(bic_a, dict):
        bic_a = bic_a["bic_features"]
    if isinstance(bic_b, dict):
        bic_b = bic_b["bic_features"]

    array_a = np.array(bic_a)
    array_b = np.array(bic_b)

    cos = cos_vectors(array_a, array_b)
    return cos


def similarity_bic_dlog(frame_a, frame_b):

    bic_a = cv3.bic(frame_a, 32)
    bic_b = cv3.bic(frame_b, 32)

    # pega o vetor de caracteristicas do bic se for um dict
    if isinstance(bic_a, dict):
        bic_a = bic_a["bic_features"]
    if isinstance(bic_b, dict):
        bic_b = bic_b["bic_features"]

    array_a = np.array(bic_a)
    array_b = np.array(bic_b)

    return dlog(array_a, array_b)


def compare_times(video, csv_dict, lst_frames):
    """
        Compara os tempos da lista de frames de corte com o csv da planilha

        retorna a acurácia
    """

    def compare(ele, min_value, max_value):
        time = ele["time"].split(".")[0]
        time = dt.strptime(time, "%H:%M:%S")
        if (
            (time.second + 1) == max_value["second"]
            and time.minute == max_value["minute"]
        ) or (
            (time.second - 1) == min_value["second"]
            and time.minute == min_value["minute"]
        ):
            return True
        else:
            return False

    corrects = 0

    for obs in csv_dict:
        time = obs["timestamp"]
        time = dt.strptime(time, "%H:%M:%S")
        min_value = {"second": time.minute - 1, "minute": time.hour}
        max_value = {"second": time.minute + 1, "minute": time.hour}

        values = list(filter(lambda x: compare(x, min_value, max_value), lst_frames))
        pprint(values)
        corrects += len(values)

    accuracy = corrects / len(csv_dict)
    print("corrects", corrects)
    print("len(csv_dict)", len(csv_dict))
    print("accuracy", accuracy)
    return accuracy


@nb.jit
def shot_boundary_detection(video, lst_frames, function, limit=1):
    """
    Função para manipular a lista contendo os frames extraidos de um video, e detectar entre quais deles ocorre um corte
    :param lst_frames: lista de dicts contendo ids dos frames e o frame em si, sendo as keys = frames_id e frames
    :param function: função de similaridade para comparação entre dois frames
    :param limit: fator que determinará se o resultado da função e comparação reflete um corte
    :return: uma lista de dicts que estão dentro do limite especificado pelo limiar
    """
    lst_shot_boundary = []
    fa = lst_frames[0]  # frameA
    percent = 0
    for frame in lst_frames[1:]:
        val = function(fa["frame"], frame["frame"])

        # passou do limiar, corte detectado
        if val <= limit:
            lst_shot_boundary.append(
                {
                    "frame_id_A": fa["frame_id"],
                    "frame_id_B": frame["frame_id"],
                    "time": get_time_frame(video, frame["frame_id"]),
                    "similarity": val,
                }
            )
        fa = frame
        percent += 1
        print("# concluído - %d/%d" % (percent, len(lst_frames)))
    return lst_shot_boundary


@nb.jit
def shot_boundary_detection_grid(
    video, lst_frames, function, limit=1, nparts=5, mask=[]
):

    # checando a integridade da mascara
    if len(mask) is 0:
        mask = []
        for i in range(nparts * nparts):
            mask.append(1)
    elif len(mask) < nparts * nparts:
        raise Exception("tamanho da mascara não compreende o tamanho de tiles do grid")

    if len(lst_frames) is 0:
        raise Exception("Lista de frames vazia!")

    lst_shot_boundary = []
    fa = lst_frames[0]

    # para cada frame geram-se os tiles da img
    for frame in lst_frames[1:]:
        tiles_fa = cv3.generate_tiles(fa.get("frame"), size=nparts)
        tiles_fb = cv3.generate_tiles(frame.get("frame"), size=nparts)

        print(fa["frame_id"], frame["frame_id"])

        # para cada tile é feita a similaridade entre eles
        value_sim_tiles = []
        for idx_tile, tile_fb in enumerate(tiles_fb):
            tile_fa = tiles_fa[idx_tile]

            # calcula a similaridade entre os tiles do FA com os do FB
            value_sim_tile = function(tile_fa, tile_fb)

            value_sim_tiles.append(value_sim_tile)

        value_sim_tiles = np.array(value_sim_tiles)
        # multiplica os pesos da máscara com os tiles para dar maior significância
        value_sim_tiles = np.multiply(value_sim_tiles, mask)

        # soma os valores e os pesos
        sum_values = np.sum(value_sim_tiles)
        sum_weight = np.sum(mask)

        # calcula a média ponderada
        mean = sum_values / sum_weight

        if mean <= limit:
            lst_shot_boundary.append(
                {
                    "frame_id_A": fa["frame_id"],
                    "frame_id_B": frame["frame_id"],
                    "similarity": mean,
                    "time": get_time_frame(video, frame["frame_id"]),
                }
            )
        fa = frame
    return lst_shot_boundary


def save_frames(video, path, lst_frames, prefix="frame", sufix="jpg"):
    """
        Dada uma lista de frames, salva-a em um determinado caminho.
    """
    for frame in lst_frames:
        frame_id = frame.get("frame_id")
        time = get_time_frame(video, frame_id)

        name_arq = "%s_%d_%s.%s" % (prefix, frame_id, time, sufix)
        cv3.save_img(path, name_arq, frame["frame"])
    return True
