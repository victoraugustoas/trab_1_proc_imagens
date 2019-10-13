import platform
from datetime import datetime as dt
from datetime import timedelta
from pprint import pprint

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


def load_videos_janela(video_file, cut=25):
    """
    essa função abre o video e retorna os frames multiplos de 10
    :param video_file: arquivo de video que deve ser aberto, para que os frames sejam capturados
    :param cut: fator de corte para os frames, ou seja, quantidade de fps para criarmos a janela
    """
    # extrair alguns frames e salvar, dps gerar o histograma deles

    # print "load_videos"
    capture = cv2.VideoCapture(video_file)

    read_flag, frame = capture.read()
    vid_frames = []
    i = 1
    # print read_flag
    janela = 2 * cut + 1
    frames_janela = []

    # ttl_janelas = 0
    # retorno = []

    while read_flag:
        # print i
        if i <= janela:
            frames_janela.append(frame)
        else:
            # a janela fechou
            # print(janela)
            # print(int(len(frames_janela)/2))
            vid_frames.append(
                frames_janela[int(len(frames_janela) / 2)]
            )  # pegando apenas o frame do meio da janela

            frames_janela = [frame]  # limpando a janela e colocado o frame atual nela
            janela += 2 * cut + 1
            # print frame.shape
        read_flag, frame = capture.read()
        i += 1
    vid_frames = np.asarray(vid_frames, dtype="uint8")[:-1]
    capture.release()
    print("total de frames:", i)
    print("total de frames selecionados", len(vid_frames))
    return vid_frames


def load_video(video_file):
    """
    função para extrair os frames de um video, nessa função especificamente serão retornado os frames em um intervalo
    de 10, ou seja, a cada 10 frames 1 será retornado.
    Isso é importante pois não foi possível retornar todos os frames, ele crasha
    :param video_file: video que deve ser aberto para que seja extraido seus frames
    :return: um lista contendo os frames, eles já estão prontos para serem exibidos
    """
    # print "load_videos"
    capture = cv2.VideoCapture(video_file)

    read_flag, frame = capture.read()
    vid_frames = []
    i = 1
    # print read_flag

    while read_flag:
        # print i
        if i % 25 == 0:
            vid_frames.append(frame)
            #                print frame.shape
        read_flag, frame = capture.read()
        i += 1
    vid_frames = np.asarray(vid_frames, dtype="uint8")[:-1]
    # print 'vid shape'
    # print vid_frames.shape
    capture.release()
    print(i)
    return vid_frames


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
def similarity_hist_global(frame_a, frame_b):
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

    # caso a qtd de cores não seja igual
    if len(bic_a.get("pallet")) != len(bic_b.get("pallet")):
        return 0.0

    # monta o vetor de caracteristicas do bic
    array_a = bic_a["inner"] + bic_a["out"]
    array_b = bic_b["inner"] + bic_b["out"]

    array_a = np.array(array_a)
    array_b = np.array(array_b)

    cos = cos_vectors(array_a, array_b)
    return cos


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
        corrects += len(values)

    accuracy = corrects / len(csv_dict)
    return accuracy


def tester():
    # esse é um trecho do meu tester, que eu tava usando para fazer a similaridade entres os frames
    # play_video("../videos/bf42.mp4")
    # extraido de : https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
    # link da documentação: https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calchist
    """frames = vito.load_video("../videos/bf42.mp4")
    i = 0
    for f in frames[:25]:
        vito.save_img("../imagens/frames", "frames" + str(i) + ".png", f)
        #cv2.imshow("f1", f)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        i+= 1
    quit()"""

    image1 = cv2.imread("../imagens/frames/frames0.png")
    image2 = cv2.imread("../imagens/frames/frames5.png")

    hist1 = cv2.calcHist(
        [image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist(
        [image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )
    hist2 = cv2.normalize(hist2, hist2).flatten()

    print("simi entre 2 frames:", cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT))


def detecta_corte_grid(video, lista_frames, limiar, nparts, mask=[]):
    """
    função para detectar corte entre os frames de uma lista, porém analisaremos os grids da imagem, ou seja, compararemos
        a similaridade por meio do quão parecidos são os grids
    :param lista_frames: lista de dicts contendo ids dos frames e o frame em si, sendo as keys = frames_id e frames
    :param limiar: fator que determinará se o resultado da função e comparação reflete um corte
    :param nparts: numero de partições do grid
    :param mask: vetor contendo os pesos que devem ser dados as posições do grid
    :return: uma lista de tuplas contendo a comparação feita e o valor resultante da comparação, porém só será retornado
            as comparações que forem acima do limiar, ou seja, só serão retornados os cortes
    """
    lista_cortes = []

    # checando a integridade da mascara
    if len(mask) is 0:
        mask = []
        for i in range(nparts * nparts):
            mask.append(1)
    elif len(mask) < nparts * nparts:
        raise Exception("tamanho da mascara não compreende o tamanho de tiles do grid")

    # tirando o 1º frame na mão
    fa = lista_frames[0]  # frameA
    # tirando o histograma local nas 3 bandas das partições
    tiles_fa = cv3.histogram_local(fa["frame"], nparts)

    # iterando sobre os outros frames
    for frame in lista_frames[1:]:
        # tirando o histograma local nas 3 bandas do frame b
        tiles_frame = cv3.histogram_local(frame["frame"], nparts)

        lista_a = []
        lista_b = []
        tiles_simi = []
        # percorrendo todos os grids
        for tile in range(0, nparts * nparts):
            for channel in range(0, 3):
                for color in range(0, 256):
                    lista_a.append(tiles_fa[tile][channel][color])
                    lista_b.append(tiles_frame[tile][channel][color])
            # tirando a similaridade entre os tiles do grid
            array_a = np.array(lista_a)
            array_b = np.array(lista_b)
            cos = cos_vectors(array_a, array_b)
            tiles_simi.append(
                cos
            )  # criando um vetor com as similaridades entre os tiles

        # vetor contendo a similaridade entre os tiles já multiplicados os pesos
        similaridades = np.multiply(tiles_simi, mask)
        soma_valores = np.sum(similaridades)
        soma_pesos = np.sum(mask)

        # todo arrumar uma mas boa
        media = soma_valores / soma_pesos

        if media > limiar:  # corte
            lista_cortes.append(
                {
                    "frame_id_A": fa["frame_id"],
                    "frame_id_B": frame["frame_id"],
                    "similarity": media,
                    "time": get_time_frame(video, frame["frame_id"]),
                }
            )

        # resetando para a proxima iteraçao
        fa = frame
        tiles_fa = tiles_frame
    return lista_cortes


@nb.jit
def shot_boundary_detection(video, lst_frames, function, limit=0.01):
    """
    Função para manipular a lista contendo os frames extraidos de um video, e detectar entre quais deles ocorre um corte
    :param lst_frames: lista de dicts contendo ids dos frames e o frame em si, sendo as keys = frames_id e frames
    :param function: função de similaridade para comparação entre dois frames
    :param limit: fator que determinará se o resultado da função e comparação reflete um corte
    :return: uma lista de dicts que estão dentro do limite especificado pelo limiar
    """
    lst_shot_boundary = []
    fa = lst_frames[0]  # frameA
    for frame in lst_frames[1:]:
        pprint(fa)
        pprint(frame)
        pprint("function")
        val = function(fa["frame"], frame["frame"])

        # passou do limiar, corte detectado
        if val > limit:
            lst_shot_boundary.append(
                {
                    "frame_id_A": fa["frame_id"],
                    "frame_id_B": frame["frame_id"],
                    "time": get_time_frame(video, frame["frame_id"]),
                    "similarity": val,
                }
            )
        fa = frame
    return lst_shot_boundary
