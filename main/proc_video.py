import cv2
import numba as nb
import numpy as np
import datetime as dt

import proc_img as cv3


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

    return str(dt.timedelta(milliseconds=ms))


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
def get_frames(video, fps):
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
            info = {"frame_id": i - 1, "frame": frame}
            vid_frames.append(info)

        read_flag, frame = video.read()
        i += 1

    return vid_frames


@nb.jit
def similarity(frame_a, frame_b):

    """
        Calcula a similaridade entre dois frames utilizando a distancia euclidiana

        retorna a distancia
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

    produto_interno = np.inner(array_a, array_b)
    norm_a = np.linalg.norm(array_a)
    norm_b = np.linalg.norm(array_b)

    cos = produto_interno / (norm_a * norm_b)

    return cos


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


def detecta_corte(lista_frames, limiar=0.01):
    """
    Função para manipular a lista contendo os frames extraidos de um video, e detectar entre quais deles ocorre um corte
    :param lista_frames: lista de dicts contendo ids dos frames e o frame em si, sendo as keys = frames_id e frames
    :param limiar: fator que determinará se o resultado da função e comparação reflete um corte
    :return: uma lista de tuplas contendo a comparação feita e o valor resultante da comparação, porém só será retornado
            as comparações que forem acima do limiar, ou seja, só serão retornados os cortes
    """
    lista_cortes = []
    fa = lista_frames[0]  # frameA
    for frame in lista_frames[1:]:
        # print(fA["frame_id"], "-->", frame["frame_id"])
        val = similarity(fa["frame"], frame["frame"])

        # passou do limiar, corte detectado
        if val > limiar:
            lista_cortes.append(
                {
                    "frame_id_A": fa["frame_id"],
                    "frame_id_B": frame["frame_id"],
                    "similarity": val,
                }
            )
        fa = frame
    return lista_cortes


def detecta_corte_grid(lista_frames, limiar, nparts):
    """
    :param lista_frames: lista de dicts contendo ids dos frames e o frame em si, sendo as keys = frames_id e frames
    :param limiar: fator que determinará se o resultado da função e comparação reflete um corte
    :param nparts: numero de partições do grid
    :return: uma lista de tuplas contendo a comparação feita e o valor resultante da comparação, porém só será retornado
            as comparações que forem acima do limiar, ou seja, só serão retornados os cortes
    """
    lista_cortes = []
    fa = lista_frames[0]  # frameA
    #tirando o histograma local nas 3 bandas das partições
    tiles_fa = (cv3.histogram_local(fa["frame"], nparts))
    print(len(tiles_fa), len(tiles_fa[0]), len(tiles_fa[0][0]))


    for frame in lista_frames[1:]:
        #tirando o histograma local nas 3 bandas do frame b
        tiles_frame = (cv3.histogram_local(frame["frame"], nparts))

        lista_a = []
        lista_b = []
        #percorrendo todos os grids
        for i in range(0, nparts*nparts):
            for j in range(0, 3):
                for k in range(0, 256):
                    lista_a.append(tiles_fa[i][j][k])
                    lista_b.append(tiles_frame[i][j][k])
            array_a = np.array(lista_a)
            array_b = np.array(lista_b)
            print(i, np.linalg.norm(array_a - array_b))
            lista_a = []
            lista_b = []

        #todo codar comparação entre os tiles com uma mask
        #todo mask personalizavel ?
        #todo codar retorno formato vito
    return True