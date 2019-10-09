import os

import warnings

from math import trunc

import cv2

import numpy as np

import matplotlib.pyplot as plt

from random import randint
from math import sqrt, pow
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import numba as nb

# remove os warnings eventuais
warnings.simplefilter("ignore")


def open_img(path, gray=False):
    """
        Abre a imagem

        path: local da imagem
        gray: se a imagem será aberta em tons de cinza
    """
    img = cv2.imread(
        path, cv2.IMREAD_GRAYSCALE if gray == True else cv2.IMREAD_UNCHANGED
    )
    img = img.reshape(status_img(img))
    return img


def save_img(path, name_arq, matrix):
    """
        Salva a imagem

        path: local onde será salvo
        name_arq: nome do arquivo
        matrix: obj da img

        gray: true
    """
    path = os.path.join(path, name_arq)
    return cv2.imwrite(path, matrix)


def status_img(matrix):
    """
        Recebe o obj da img

        retorna o numero de linhas, colunas e canais da img

        retorno (nrows, ncols, channels) se imagem colorida
        retorno (nrows, ncols, 1) se imagem em escala de cinza

        gray: true
    """
    if len(matrix.shape) > 2:
        return (matrix.shape[0], matrix.shape[1], matrix.shape[2])
    else:
        return (matrix.shape[0], matrix.shape[1], 1)


def generate_histogram(
    values,
    title,
    name_file,
    color,
    names=np.arange(0, 256),
    path="./data",
    xlabel="Valores de cinza",
    ylabel="Frequência",
):
    """
        Gere o histograma de um vetor

        names: valores do eixo x
        values: valores correspondentes aos names
        title: titulo do gráfico
        name_file: nome do arquivo gerado,
        color: cor das barras,
        path: local onde será salvo,
        xlabel: legenda para eixo x
        ylabel: legenda para eiyo y
    """
    plt.bar(names, values, color=color)
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(path, name_file))
    return plt.close()


@nb.jit
def brightness(matrix, coefficient):
    """
        Recebe o obj da imagem e um coeficiente

        Aplica uma soma dos pixels da imagem com o coeficiente
        para aumentar o brilho

        gray: true
    """
    nrows, ncols, channels = status_img(matrix)

    # copia a matriz
    matrix = matrix.copy()

    for i in range(nrows):
        for j in range(ncols):

            # para os canais RGB
            for channel in range(channels):
                sum_ = coefficient + matrix[i][j][channel]
                if sum_ >= 255:
                    matrix[i][j][channel] = 255
                elif sum_ <= 0:
                    matrix[i][j][channel] = 0
                else:
                    matrix[i][j][channel] = sum_

    return matrix


@nb.jit
def negative(matrix):
    """
        Recebe o obj de imagem e transforma a imagem em negativo

        gray: true
    """
    nrows, ncols, channels = status_img(matrix)

    matrix = matrix.copy()

    for i in range(nrows):
        for j in range(ncols):
            for channel in range(channels):
                matrix[i][j][channel] = 255 - matrix[i][j][channel]

    return matrix


def generate_histograms(hist_blue, hist_green, hist_red, path, prefix=""):
    """
        Recebe 3 vetores:
            hist_blue,
            hist_green,
            hist_red
        representando os histogramas de uma imagem

        Gera 3 gráficos de histogramas, 1 para cada vetor
        
        path: local de destino
        prefix: prefixo do nome do arquivo
    """

    # gera os histogramas
    generate_histogram(
        hist_blue,
        "Histograma Global - Azul",
        prefix + "histograma_global_blue.jpg",
        "blue",
        path=path,
    )
    generate_histogram(
        hist_green,
        "Histograma Global - Verde",
        prefix + "histograma_global_green.jpg",
        "green",
        path=path,
    )
    generate_histogram(
        hist_red,
        "Histograma Global - Vermelho",
        prefix + "histograma_global_red.jpg",
        "red",
        path=path,
    )

    return True


@nb.jit
def histogram_global(matrix):
    """
        Calcula o histograma global da imagem e retorna 3 vetores

        matrix: obj da img

        retorna (hist_blue, hist_green, hist_red) para img com 3 bandas
        retorna (hist_blue) para img com 1 banda

        gray: true
    """
    nrows, ncols, channels = status_img(matrix)

    matrix = matrix.copy()

    hist_blue = np.zeros(256)
    hist_green = np.zeros(256)
    hist_red = np.zeros(256)

    for i in range(nrows):
        for j in range(ncols):
            for channel in range(channels):

                pixel_value = int(matrix[i][j][channel])

                # canal blue
                if channel == 0:
                    hist_blue[pixel_value] += 1
                elif channel == 1:
                    hist_green[pixel_value] += 1
                else:
                    hist_red[pixel_value] += 1
    if channels == 1:
        return hist_blue
    else:
        return (hist_blue, hist_green, hist_red)


def histogram_local(matrix, nparts, channel=0):
    """
        Recebe o obj da img

        nparts: qtd de partições da imagem (n*n)
        
        channel: valor do canal b=1, g=2, r=3

        retorna uma lista com vetores de histograma
    """

    partition = generate_tiles(matrix, nparts)

    hist_parts = []

    for i, part in enumerate(partition):
        if channel:
            hist_parts.append(histogram_global(part)[channel - 1])
        else:
            hist_parts.append(histogram_global(part))

    return hist_parts


def save_vector(path, *vectors, prefix="", name=""):
    """
        Recebe vetores e concatena-os

        path: local de destino
        prefix: prefixo do nome do arquivo
        *vectors: vetores a serem concatenados
    """

    vector_concat = []
    for vector in vectors:
        for ele in vector:
            try:
                vector_concat.extend(ele)
            except:
                vector_concat.append(ele)

    with open(os.path.join(path, prefix + name), "w") as arq:
        arq.write("tamanho: %s\n" % (len(vector_concat)))
        arq.write(str(vector_concat))


# sem paralelização é melhor
def generate_position(max_lin, max_col):
    """
    dado dois numeros maximos, irá gerar dois outros numeros aleatorios entre esses dois maximos, ou seja
    ira gerar um pixel aleatorio dentro dos limites da imagem
    :param max_lin: numero maxixmo de linhas
    :param max_col: numero maximo de colunas
    :return: retorna uma tupla contendo um numero de lin e um de coluna, tuple(lin, col)
    """
    # discontando 1 pq ele gera o limite superior
    aux = (randint(0, max_lin - 1), (randint(0, max_col - 1)))
    return tuple(aux)


# sem paralelização é melhor
def generate_noise(matrixInit, percent=10, noise="salt"):
    """
    dada uma imagem, irá inserir ruido nela, aleatóriamente, obedecendo uma porcentagem.

    :param matrixInit: a imagem que se deseja adicionar ruido tipo sal

    :param percent: a porcentagem de ruido que se deseja aplicar na imagem (valor inteiro)

    :param noise: variavel que carrega o tipo de ruido, "salt" = branco, "pepper" = preto

    :return: a matrix referente a imagem, porém acrescido de ruido

    gray: true
    """
    matrix = matrixInit.copy()
    if noise == "salt":
        noise = 255
    elif noise == "pepper":
        noise = 0
    else:
        raise Exception("Tipo de ruído incorreto -> " + noise)

    if type(percent) == int:
        percent = percent / 100
    elif type(percent) == float and percent <= 1:
        pass
    else:
        raise Exception("O número informado é um float maior que 1")

    lista_pixel = []
    nLins, nCols, canais = status_img(matrix)
    qntd_pixels = nLins * nCols

    dict_pixel = {}
    # gerando posições aleatorias até que o limitar percentual seja atingido
    while len(lista_pixel) < int(qntd_pixels * percent):
        pixel = generate_position(nLins, nCols)
        aux = str(
            pixel
        )  # guaradaremos os pixels como strings só pra consultar de forma mais eficiente
        if dict_pixel.get(aux, 0) == 0:
            # se entraou aqui é pq nao existe no dict
            dict_pixel[aux] = 1  # coloca no dict
            lista_pixel.append(pixel)  # coloca na lista

    for pixel in lista_pixel:
        for canal in range(canais):
            matrix[pixel[0]][pixel[1]][canal] = noise

    return matrix


@nb.jit
def filtro_media(matrixInit, iterations=1):
    """
    aplica o filtro da media nos pixels da imagem, com uma janela 3x3
    
    :param matrixInit: imagem a qual deve ser aplicado o filtro
    
    :param iterations: valor das k iterações

    :return: copia da imagem com o filtro aplicado

    gray: true
    """

    matrix = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)
    janela = 3  # sempre 3x3
    # não percorremos a coluna 0, nem a ultima coluna
    # nao percorreremos a linha 0 nem a ultima linha

    # os vizinhos serão guardados em um vetor na seguinte ordem: sentido horário a partir da celula superior ao pixel
    # ou seja a ordem, central, norte, nordeste, leste, sudeste e assim por diante. Nessa organização é ficilitada a aplicação de
    # pesos
    for k in range(iterations):
        for i in range(1, nLins - 1):
            for j in range(1, nCols - 1):
                for ch in range(canais):
                    pixels = []
                    pixels.append(matrix[i][j][ch])  # central
                    pixels.append(matrix[i - 1][j][ch])  # norte
                    pixels.append(matrix[i - 1][j + 1][ch])  # nordeste
                    pixels.append(matrix[i][j + 1][ch])  # leste
                    pixels.append(matrix[i + 1][j + 1][ch])  # sudeste
                    pixels.append(matrix[i + 1][j][ch])  # sul
                    pixels.append(matrix[i + 1][j - 1][ch])  # sudoeste
                    pixels.append(matrix[i][j - 1][ch])  # oeste
                    pixels.append(matrix[i - 1][j - 1][ch])  # noroeste

                    soma = 0
                    for pix in pixels:
                        soma += pix
                    soma = int(soma / 9)

                    if soma <= 255:
                        matrix[i][j][ch] = soma
                    else:
                        matrix[i][j][ch] = 255
    return matrix


@nb.jit
def filtro_moda(matrixInit, iterations=1):
    """
    essa função aplica o filtro da moda na imagem dada como entrada
    
    :param matrixInit: a matriz referente a imagem a ser processada
    
    :param iterations: numero de iterações a ser aplicado na imagem

    :return: a matriz processada pela função

    gray: true
    """
    matrix = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)
    janela = 3  # sempre 3x3
    # não percorremos a coluna 0, nem a ultima coluna
    # nao percorreremos a linha 0 nem a ultima linha

    # os vizinhos serão guardados em um vetor na seguinte ordem: sentido horário a partir da celula superior ao pixel
    # ou seja a ordem, central, norte, nordeste, leste, sudeste e assim por diante

    for k in range(iterations):
        for i in range(1, nLins - 1):
            for j in range(1, nCols - 1):
                for ch in range(canais):
                    pixels = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    pixels[0] = matrix[i][j][ch]  # central
                    pixels[1] = matrix[i - 1][j][ch]  # norte
                    pixels[2] = matrix[i - 1][j + 1][ch]  # nordeste
                    pixels[3] = matrix[i][j + 1][ch]  # leste
                    pixels[4] = matrix[i + 1][j + 1][ch]  # sudeste
                    pixels[5] = matrix[i + 1][j][ch]  # sul
                    pixels[6] = matrix[i + 1][j - 1][ch]  # sudoeste
                    pixels[7] = matrix[i][j - 1][ch]  # oeste
                    pixels[8] = matrix[i - 1][j - 1][ch]  # noroeste

                    a = np.array(pixels)
                    counts = np.bincount(a)
                    moda = np.argmax(counts)

                    matrix[i][j][ch] = int(moda)
    return matrix


@nb.jit
def filtro_mediana(matrixInit, iterations=1):
    """
    essa função aplica o filtro da mediana na imagem dada como entrada

    :param matrixInit: a matriz referente a imagem a ser processada

    :param iterations: o numero de iterações que o filtro deve ser aplicado

    :return: a matriz processada pela função

    gray: true
    """
    matrix = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)
    janela = 3  # sempre 3x3
    # não percorremos a coluna 0, nem a ultima coluna
    # nao percorreremos a linha 0 nem a ultima linha

    # os vizinhos serão guardados em um vetor na seguinte ordem: sentido horário a partir da celula superior ao pixel
    # ou seja a ordem, central, norte, nordeste, leste, sudeste e assim por diante
    for k in range(iterations):
        for i in range(1, nLins - 1):
            for j in range(1, nCols - 1):
                for ch in range(canais):
                    pixels = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    pixels[0] = matrix[i][j][ch]  # central
                    pixels[1] = matrix[i - 1][j][ch]  # norte
                    pixels[2] = matrix[i - 1][j + 1][ch]  # nordeste
                    pixels[3] = matrix[i][j + 1][ch]  # leste
                    pixels[4] = matrix[i + 1][j + 1][ch]  # sudeste
                    pixels[5] = matrix[i + 1][j][ch]  # sul
                    pixels[6] = matrix[i + 1][j - 1][ch]  # sudoeste
                    pixels[7] = matrix[i][j - 1][ch]  # oeste
                    pixels[8] = matrix[i - 1][j - 1][ch]  # noroeste

                    mediana = np.median(pixels)
                    matrix[i][j][ch] = int(mediana)
    return matrix


@nb.jit
def hist_to_img(matrix, hist):
    """
        Coloca os valores do histograma na imagem
        
        matrix: matriz em escala de cinza

        hist: histograma com os valores
    """

    nrows, ncols, channels = status_img(matrix)

    matrix = matrix.copy()

    for i in range(nrows):
        for j in range(ncols):
            matrix[i][j][0] = hist[matrix[i][j][0]]

    return matrix[:, :, 0]


def equalize_hist(matrix, hist_vect):
    """
        Equaliza o histograma de uma imagem em escala de cinza

        matrix: obj da img
        hist_vect: histograma da imagem em uma banda
    """
    nrows, ncols, channels = status_img(matrix)

    # realiza uma cópia da variável para n alterar a mesma
    hist_vect = hist_vect.copy()

    total_pixels = nrows * ncols

    # calcula a probabilidade de cada cor no histograma
    for i, ele in enumerate(hist_vect):
        hist_vect[i] = hist_vect[i] / total_pixels

    # calcula a probabilidade acumulativa
    for i in range(1, len(hist_vect)):
        hist_vect[i] += hist_vect[i - 1]

    # normaliza os valores de volta para o espaço de 256 cores
    for i, ele in enumerate(hist_vect):
        hist_vect[i] *= 255

        # pois os valores de cinza são inteiros
        hist_vect[i] = trunc(hist_vect[i])

    return hist_vect


def generate_tiles(matrix, size=5):
    """
        Retorna um array com as partições da imagem

        matrix: obj da img
        size: tamanho da grid (n*n)
    """
    nrows, ncols, channels = status_img(matrix)

    chunk_row = trunc(nrows / size)
    chunk_col = trunc(ncols / size)

    tiles = []

    for i in range(size):
        for j in range(size):
            part = matrix[
                (i * chunk_row) : (i * chunk_row) + chunk_row,
                (j * chunk_col) : (j * chunk_col) + chunk_col,
                :,
            ]
            tiles.append(part)

    return tiles


def join_tiles(tiles, size=5):
    """
        Junta todos as partes da img em uma matriz

        tiles: array com as partes da img
        size: tamanho da grid (n*n)
    """
    aux_matrix = []
    for j in range(size):
        j = j * size
        aux_arr = np.concatenate(tiles[j : j + size], axis=1)
        aux_matrix.append(aux_arr)
    aux_matrix = np.concatenate(aux_matrix[:], axis=0)

    return aux_matrix


def recreate_img(colors, matrix_to_array, nrows, ncols):
    """
        Recria a matriz de imagem, com base nas cores e no vetor da imagem

        colors: vetor de cores com X cores
        matrix_to_array: vetor de cores da imagem que sofreu quantização
        nrows: # de linhas da img resultante
        ncols: # de colunas da img resultante
    """
    # colors é um vetor de tamanho X de 3 dimensões se colorida, 1 dimensão se cinza
    channels = colors.shape[1]

    # cria uma matriz de zeros com dimensões da img
    img = np.zeros((nrows, ncols, channels))

    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            img[row][col] = colors[matrix_to_array[idx]]
            idx += 1
    return img


def quantization_colors(matrix, color=32):
    """
        Retorna a imagem com redução de cor

        matrix: obj da img
        color: qtd de cores presentes na imagem resultante
    """
    matrix = matrix.copy()

    nrows, ncols, channels = status_img(matrix)

    # transforma a matriz em um vetor
    matrix_to_array = np.reshape(matrix, (nrows * ncols, channels))

    # para fazer o treinamento do Kmeans, pegamos 1000 elementos aleatórios do vetor de img
    image_array_sample = shuffle(matrix_to_array, random_state=0)[:1000]

    # fazemos o treinamento do Kmeans com o array de treinamento
    kmeans = KMeans(n_clusters=color, random_state=0).fit(image_array_sample)

    # vetor com K cores de tamanho nrows*ncols
    matrix_color_quantizated = kmeans.predict(matrix_to_array)

    # os clusters do Kmeans serão as cores
    colors = kmeans.cluster_centers_

    # traz os valores para inteiro
    for i, color_vec in enumerate(colors):
        for j, color in enumerate(color_vec):
            colors[i][j] = int(colors[i][j])

    return recreate_img(colors, matrix_color_quantizated, nrows, ncols)


def arrays_equal(*arrays):
    bool_verification = True
    for idx in range(1, len(arrays)):
        bool_verification = (
            np.array_equal(arrays[idx - 1], arrays[idx]) and bool_verification
        )
    return bool_verification


# SETADO PARA 1 CANAL APENAS
def bic(matrixInit, qntCores):
    """
    Essa função aplicará a imagem de entrada o descritor de cor bic, que dirá se se os pixels são de borda ou de interior
    essa função retornará a imagem com os pixels transformados
    :param matrixInit: imagem de entrada, já quantizada
    :return: um vetor contendo a contagem da intensidade de cor dos pixel nos niveis high e low, sendo a primeira posição a paleta de cores
    a segunda posição o vetor de high e a segunda posição o vetor de low
    """
    matrix = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)

    dict_cor_high = {}
    dict_cor_low = {}

    for i in range(1, nLins - 1):
        for j in range(1, nCols - 1):
            # pixels = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            central = matrix[i][j]  # central
            norte = matrix[i - 1][j]  # norte
            direita = matrix[i][j + 1]  # leste
            sul = matrix[i + 1][j]  # sul
            esquerda = matrix[i][j - 1]  # oeste

            if arrays_equal(central, norte, direita, sul, esquerda):
                # central é interior
                # matrix[i][j] = 255
                pix = matrix[i][j]

                if dict_cor_high.get(str(pix), 0) == 0:
                    dict_cor_high[str(pix)] = 1
                else:
                    dict_cor_high[str(pix)] += 1
            else:
                # central é borda
                # matrix[i][j] = 0
                pix = matrix[i][j]

                if dict_cor_low.get(str(pix), 0) == 0:
                    dict_cor_low[str(pix)] = 1
                else:
                    dict_cor_low[str(pix)] += 1

    # monta a paleta de cores da imagem
    meta = []
    idx = 0
    for name in dict_cor_high.keys():
        filtered = filter(lambda x: x.get("color") == name, meta)
        if len(list(filtered)) == 0:
            meta.append({"color": name, "index": idx})
            idx += 1

    for name in dict_cor_low.keys():
        filtered = filter(lambda x: x.get("color") == name, meta)
        if len(list(filtered)) == 0:
            meta.append({"color": name, "index": idx})
            idx += 1

    response_interno = [None for x in meta]
    for info in meta:
        color = info.get("color")
        idx = info.get("index")

        if dict_cor_high.get(color):
            response_interno[idx] = dict_cor_high.get(color)

    response_externo = [None for x in meta]
    for info in meta:
        color = info.get("color")
        idx = info.get("index")

        if dict_cor_low.get(color):
            response_externo[idx] = dict_cor_low.get(color)

    return [meta, response_interno, response_externo]


def filtro_sobel(matrixInit):
    """
    :param matrixInit: matrix inicial EM TONS DE CINZA  e quantizada passada por parametro, ou seja é a imagem ao
                       qual o filtro será aplicado
    :return: matrix processada com as bordas realçadas
    """
    matrix = matrixInit.copy()
    matrixX = matrixInit.copy()
    matrixY = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)

    for i in range(1, nLins - 1):
        for j in range(1, nCols - 1):
            a = int(matrix[i - 1][j - 1][0])  # noroeste
            b = int(matrix[i - 1][j][0])  # norte
            c = int(matrix[i - 1][j + 1][0])  # nordeste
            d = int(matrix[i][j - 1][0])  # oeste
            e = int(matrix[i][j][0])  # central
            f = int(matrix[i][j + 1][0])  # leste
            g = int(matrix[i + 1][j - 1][0])  # sudoeste
            h = int(matrix[i + 1][j][0])  # sul
            aux = int(matrix[i + 1][j + 1][0])  # sudeste

            Sx1 = c + 2 * f + aux
            Sx2 = a + 2 * d + g
            Sx = Sx1 - Sx2
            matrixX[i][j][0] = Sx

            Sy1 = g + 2 * h + aux
            Sy2 = a + 2 * b + c
            Sy = Sy1 - Sy2
            matrixY[i][j][0] = Sy

    for i in range(nLins - 1):
        for j in range(nCols - 1):
            Sx = matrixX[i][j][0]
            Sy = matrixY[i][j][0]

            Sx = int(pow(Sx, 2))
            Sy = int(pow(Sy, 2))
            matrix[i][j][0] = int(sqrt(Sx + Sy))
    return matrix


def linear_enhancement(matrix, a, b):
    """
        Altera o contraste da imagem e brilho de acordo com os fatores de parametro

        matrix: obj da img
        a: inclinação da reta, altera o contraste da img
        b: altera o brilho da img
    """

    nrows, ncols, channels = status_img(matrix)
    matrix = matrix.copy()

    for row in range(nrows):
        for col in range(ncols):
            for channel in range(channels):
                pixel_value = matrix[row][col][channel]
                pixel_value = int(a * pixel_value + b)

                # realce linear
                if pixel_value >= 255:
                    pixel_value = 255

                matrix[row][col][channel] = pixel_value

    return matrix


def fatiamento(matrizInit, nv0=0, nv1=190, limiar=120):
    """
    Essa função realiza o fatiamento do histograma, de forma que se o valor de um pixel for inferior ao limiar ele será
    setado para o valor de nivel 0, caso o valor seja igual ou maior ao limiar ele será setado para o nivel 1
    :param matrizInit: imagem de entrada para o filtro, deve ser uma imagem em tom de cinza
    :param nv0: é o nivel (cor) que deve ser ajustado caso o pixel nao atenda o limiar
    :param nv1: é o nivel (cor) que deve ser ajustado caso o pixel atinja o limiar
    :param limiar: é o valor de corte que deve ser obedecido
    :return: imagem em escala de cinza com os valores ajustados
    """

    nrows, ncols, channels = status_img(matrizInit)
    if channels > 1:
        raise NotImplementedError(
            "A função só está implementada para imagens em tons de cinza!"
        )
    matriz = matrizInit.copy()

    for i in range(nrows):
        for j in range(ncols):
            if matriz[i][j] < limiar:
                matriz[i][j] = nv0
            else:
                matriz[i][j] = nv1

    return matriz


def load_videos_janela(video_file, cut=25):
    """
    essa função abre o video e retorna os frames multiplos de 10
    :param video_file: arquivo de video que deve ser aberto, para que os frames sejam capturados
    :param cut: fator de corte para os frames, ou seja, quantidade de fps para criarmos a janela
    """
    return False  # bugada
    #extrair alguns frames e salvar, dps gerar o histograma deles

    # print "load_videos"
    capture = cv2.VideoCapture(video_file)

    read_flag, frame = capture.read()
    vid_frames = []
    i = 1
    # print read_flag
    janela = 2 * cut + 1
    frames_janela = []
    ttl_janelas = 0
    retorno = []
    while (read_flag):
        # print i
        if i <= janela:
            frames_janela.append(frame)
        else: # a janela já encheu

            frames_janela = [frame]  # nova lista de frames
            ttl_janelas += 1
            janela += 2 * cut + 1
        vid_frames = np.asarray(frames_janela, dtype='uint8')[:-1]
        read_flag, frame = capture.read()
        i += 1

    capture.release()
    print("total de frames:", i)
    print("total de janelas:", ttl_janelas)
    return retorno


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

    while (read_flag):
        # print i
        if i % 25 == 0:
            vid_frames.append(frame)
            #                print frame.shape
        read_flag, frame = capture.read()
        i += 1
    vid_frames = np.asarray(vid_frames, dtype='uint8')[:-1]
    # print 'vid shape'
    # print vid_frames.shape
    capture.release()
    print(i)
    return vid_frames


def play_video(video_file):
    """
    Função para exibir o video, ela cria diversas janelas, 1 para cada frame e a fecha em seguida, dando a impressão
    de um player fixo
    :param video_file: video a ser tocado no  "player"
    """
    cap = cv2.VideoCapture(video_file)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def tester():
    # esse é um trecho do meu tester, que eu tava usando para fazer a similaridade entres os frames
    # play_video("../videos/bf42.mp4")
    #extraido de : https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
    #link da documentação: https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calchist
    '''frames = vito.load_video("../videos/bf42.mp4")
    i = 0
    for f in frames[:25]:
        vito.save_img("../imagens/frames", "frames" + str(i) + ".png", f)
        #cv2.imshow("f1", f)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        i+= 1
    quit()'''

    image1 = cv2.imread("../imagens/frames/frames0.png")
    image2 = cv2.imread("../imagens/frames/frames5.png")

    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    print("simi entre 2 frames:", cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT))
