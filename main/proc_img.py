import os

import warnings

from math import trunc

import cv2

import numpy as np

import matplotlib.pyplot as plt

from random import randint

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
    return img


def save_img(path, name_arq, matrix):
    """
        Salva a imagem

        path: local onde será salvo
        name_arq: nome do arquivo
        matrix: obj da img
    """
    path = os.path.join(path, name_arq)
    return cv2.imwrite(path, matrix)


def status_img(matrix):
    """
        Recebe o obj da img

        retorna o numero de linhas, colunas e canais da img

        retorno (nrows, ncols, channels) se imagem colorida
        retorno (nrows, ncols) se imagem em escala de cinza
    """
    try:
        return (matrix.shape[0], matrix.shape[1], matrix.shape[2])
    except:
        return (matrix.shape[0], matrix.shape[1])


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


def brightness(matrix, coefficient):
    """
        Recebe o obj da imagem e um coeficiente

        Aplica uma soma dos pixels da imagem com o coeficiente
        para aumentar o brilho
    """
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]

    # copia a matriz
    matrix = matrix.copy()

    if coefficient < 0:
        return False

    for i in range(nrows):
        for j in range(ncols):

            # para os canais RGB
            for channel in range(0, 3):
                sum_ = coefficient + matrix[i][j][channel]
                if sum_ >= 255:
                    matrix[i][j][channel] = 255
                else:
                    matrix[i][j][channel] = sum_

    return matrix


def negative(matrix):
    """
        Recebe o obj de imagem e transforma a imagem em negativo
    """
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    channels = matrix.shape[2]

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


def histogram_global(matrix):
    nrows, ncols, channels = status_img(matrix)
    hist_blue = np.zeros(256)
    hist_green = np.zeros(256)
    hist_red = np.zeros(256)

    for i in range(nrows):
        for j in range(ncols):
            for channel in range(channels):

                pixel_value = matrix[i][j][channel]

                # canal blue
                if channel == 0:
                    hist_blue[pixel_value] += 1
                elif channel == 1:
                    hist_green[pixel_value] += 1
                else:
                    hist_red[pixel_value] += 1

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


def generate_position(max_lin, max_col):
    """
    dado dois numeros maximos, irá gerar dois outros numeros aleatorios entre esses dois maximos, ou seja
    ira gerar um pixel aleatorio dentro dos limites da imagem
    :param max_lin: numero maxixmo de line
    :param max_col: numero maximo de coluna
    :return: retorna uma tupla contendo um numero de lin e um de coluna, tuple(lin, col)
    """
    # discontando 1 pq ele gera o limite superior
    aux = (randint(0, max_lin - 1), (randint(0, max_col - 1)))
    return tuple(aux)


def generate_noise(matrixInit, percent, noise):
    """
    dada uma imagem, irá inserir ruido nela (em apenas uma banda), aleatóriamente, obedecendo uma porcentagem.
    :param matrixInit: a imagem que se deseja adicionar ruido tipo sal
    :param percent: a porcentagem de ruido que se deseja aplicar na imagem (valor inteiro)
    :param noise: variavel que carrega o tipo de ruido, "salt" = branco, "pepper" = preto
    :return: a matrix referente a imagem, porém acrescido de ruido
    """
    matrix = matrixInit.copy()
    if noise == "salt":
        noise = 255
    elif noise == "pepper":
        noise = 0

    if type(percent) == int:
        percent = percent / 100
    elif type(percent) == float:
        pass

    lista_pixel = []
    nLins, nCols, canais = status_img(matrix)
    qntd_pixels = nLins * nCols
    ch = 0

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
        matrix[pixel[0]][pixel[1]][ch] = noise
        matrix[pixel[0]][pixel[1]][1] = noise
        matrix[pixel[0]][pixel[1]][2] = noise

    return matrix


def filtro_media(matrixInit, iterations=1):
    """
    aplica o filtro da media nos pixels da imagem, com uma janela 3x3
    :param matrixInit: imagem a qual deve ser aplicado o filtro
    :param iterations: valor das k iterações
    :return: copia da imagem com o filtro aplicado
    """

    matrix = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)
    ch = 0
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


def filtro_moda(matrixInit, iterations=1):
    """
    essa função aplica o filtro da moda na imagem dada como entrada
    :param matrixInit: a matriz referente a imagem a ser processada
    :param iterations: numero de iterações a ser aplicado na imagem
    :return: a matriz processada pela função
    """
    matrix = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)
    ch = 0
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
                    pixels[0] = matrixInit[i][j][ch]  # central
                    pixels[1] = matrixInit[i - 1][j][ch]  # norte
                    pixels[2] = matrixInit[i - 1][j + 1][ch]  # nordeste
                    pixels[3] = matrixInit[i][j + 1][ch]  # leste
                    pixels[4] = matrixInit[i + 1][j + 1][ch]  # sudeste
                    pixels[5] = matrixInit[i + 1][j][ch]  # sul
                    pixels[6] = matrixInit[i + 1][j - 1][ch]  # sudoeste
                    pixels[7] = matrixInit[i][j - 1][ch]  # oeste
                    pixels[8] = matrixInit[i - 1][j - 1][ch]  # noroeste

                    """aux = pd.Series(pixels)
                    moda = aux.mode().to_list()
                    moda = moda[0]"""

                    a = np.array(pixels)
                    counts = np.bincount(a)
                    moda = np.argmax(counts)

                    matrix[i][j][ch] = int(moda)
    return matrix


def filtro_mediana(matrixInit, iterations=1):
    """
    essa função aplica o filtro da mediana na imagem dada como entrada
    :param matrixInit: a matriz referente a imagem a ser processada
    :param iterations: o numero de iterações que o filtro deve ser aplicado
    :return: a matriz processada pela função
    """
    matrix = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)
    ch = 0
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

                    mediana = int(np.median(pixels))
                    matrix[i][j][ch] = mediana
    return matrix


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


def quantizacao(matrixInit, K=32):
    """
    Essa funçao pegará uma imagem com num numero N de cores e ira produzir um outro com apenas K cores
    :param matrixInit: matrix de entrada a função de quantização
    :param K: numero de cores em que a imagem deve ser clusterizada
    :return: uma matriz resultando do processo com K cores
    """
    img = matrixInit.copy()
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 25, cv2.KMEANS_RANDOM_CENTERS
    )  # RANDOM OU PP

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    return res2


def quantizacao_cinza(matrixInit, qntd_cores):
    r = 16
    print("quantidade de tons de cinza:", 255 / r)
    imgQuant = np.uint8(matrixInit / r) * r
    return imgQuant


def bic(matrixInit, qntCores):
    """
    Essa função aplicará a imagem de entrada o descritor de cor bic, que dirá se se os pixels são de borda ou de interior
    essa função retornará a imagem com os pixels transformados
    :param matrixInit: imagem de entrada, já quantizada
    :param qntCores: quantidade de cores em a imagem possui
    :return:
    """
    borda = 0
    interior = 0
    matrix = matrixInit.copy()
    nLins, nCols, canais = status_img(matrix)
    hist_high = []
    hist_low = []
    canais = 1
    dict_cor_high = {}
    dict_cor_low = {}

    for i in range(256):
        hist_high.append(0)
        hist_low.append(0)

    for i in range(1, nLins - 1):
        for j in range(1, nCols - 1):
            for ch in range(canais):
                #print(matrix[i][j][ch])
                #pixels = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                central = matrix[i][j][ch]  # central
                norte = matrix[i - 1][j][ch]  # norte
                direita = matrix[i][j + 1][ch]  # leste
                sul = matrix[i + 1][j][ch]  # sul
                esquerda = matrix[i][j - 1][ch]  # oeste

                if norte == direita == esquerda == sul:
                    # central é interior
                    #matrix[i][j][ch] = 255
                    pix = matrix[i][j][ch]
                    hist_high[pix] += 1
                    if dict_cor_high.get(str(pix), 0) == 0:
                        dict_cor_high[str(pix)] = 1
                    else:
                        dict_cor_high[str(pix)] += 1
                else:
                    # central é borda
                    #matrix[i][j][ch] = 0
                    pix = matrix[i][j][ch]
                    hist_low[pix] += 1
                    if dict_cor_low.get(str(pix), 0) == 0:
                        dict_cor_low[str(pix)] = 1
                    else:
                        dict_cor_low[str(pix)] += 1
    print("Dict High")
    print(dict_cor_high)
    print(len(dict_cor_high.keys()))

    print("Dict Low")
    print(dict_cor_low)
    print(len(dict_cor_low.keys()))
    return (hist_high, hist_low)
