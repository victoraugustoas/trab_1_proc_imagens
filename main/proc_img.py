import cv2
import os
import numpy as np
from math import trunc
import matplotlib.pyplot as plt
import warnings

# remove os warnings eventuais
warnings.simplefilter("ignore")


def open_img(path):
    img = cv2.imread(path)
    return img


def save_img(path, name_arq, matrix):
    path = os.path.join(path, name_arq)
    return cv2.imwrite(path, matrix)


def status_img(matrix):
    return (matrix.shape[0], matrix.shape[1], matrix.shape[2])


def hist_to_img(matrix, hist_blue, hist_green, hist_red):
    nrows, ncols, channels = status_img(matrix)

    for i in range(nrows):
        for j in range(ncols):
            for channel in range(channels):
                if channel == 0:
                    matrix[i][j][channel] = hist_blue[matrix[i][j][channel]]
                elif channel == 1:
                    matrix[i][j][channel] = hist_green[matrix[i][j][channel]]
                else:
                    matrix[i][j][channel] = hist_red[matrix[i][j][channel]]

    return matrix


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

    plt.bar(names, values, color=color)
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(path, name_file))
    plt.close()
    return


def brightness(matrix, coefficient):
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]

    for i in range(nrows):
        for j in range(ncols):

            # para os canais RGB
            for channel in range(0, 3):
                sum = coefficient + matrix[i][j][channel]
                if sum >= 255:
                    matrix[i][j][channel] = 255
                else:
                    matrix[i][j][channel] = sum

    return matrix


def negative(matrix):

    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    channels = matrix.shape[2]

    for i in range(nrows):
        for j in range(ncols):
            for channel in range(channels):
                matrix[i][j][channel] = 255 - matrix[i][j][channel]

    return matrix


def generate_histograms(hist_blue, hist_green, hist_red, path, prefix=""):
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

    hists = [
        (os.path.join(path, prefix + "histograma_global_blue.txt"), hist_blue),
        (os.path.join(path, prefix + "histograma_global_green.txt"), hist_green),
        (os.path.join(path, prefix + "histograma_global_red.txt"), hist_red),
    ]
    for hist in hists:
        with open(hist[0], "w") as arq:
            # converte pra inteiro os valores
            aux = list(hist[1])
            arq.write(str(aux))
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


def histogram_local(matrix, nparts, path):

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

    vector_concat = []
    for idx, hist_local in enumerate(hist_parts):
        vector_concat.extend(hist_local)

    with open(os.path.join(path, "hist_local_%s.txt" % (str(nparts))), "w") as arq:
        arq.write("tamanho: %s\n" % (len(vector_concat)))
        arq.write(str(vector_concat))

    return True
