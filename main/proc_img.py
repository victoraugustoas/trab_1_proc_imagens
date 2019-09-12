import os

import warnings

from math import trunc

import cv2

import numpy as np

import matplotlib.pyplot as plt

# remove os warnings eventuais
warnings.simplefilter("ignore")


def open_img(path):
    """
        Abre a imagem

        path: local da imagem
    """
    img = cv2.imread(path)
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

        retorno (nrows, ncols, channels)
    """
    return (matrix.shape[0], matrix.shape[1], matrix.shape[2])


def generate_histogram(
    names,
    values,
    title,
    name_file,
    color,
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

    # label do eixo x
    names = np.arange(0, 256)

    # gera os histogramas
    generate_histogram(
        names,
        hist_blue,
        "Histograma Global - Azul",
        prefix + "histograma_global_blue.jpg",
        "blue",
        path,
    )
    generate_histogram(
        names,
        hist_green,
        "Histograma Global - Verde",
        prefix + "histograma_global_green.jpg",
        "green",
        path,
    )
    generate_histogram(
        names,
        hist_red,
        "Histograma Global - Vermelho",
        prefix + "histograma_global_red.jpg",
        "red",
        path,
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


def histogram_local(matrix, nparts):
    """
        Recebe o obj da img

        nparts: qtd de partições da imagem

        retorna uma lista com vetores de histograma
    """

    nrows, ncols, channels = status_img(matrix)

    size_part = trunc(ncols / nparts)

    hist_parts = []

    for part in range(nparts):

        hist = np.zeros(256)

        start = part * size_part
        final = (part * size_part) + size_part

        for i in range(nrows):
            for j in range(start, final):
                pixel_value = matrix[i][j][0]
                hist[pixel_value] += 1

        hist_parts.append(hist)

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
