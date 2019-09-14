import os

import warnings

from math import trunc

import cv2

import numpy as np

import matplotlib.pyplot as plt

from random import randint
import pandas as pd
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

def generate_position(max_lin, max_col):
    '''
    dado dois numeros maximos, irá gerar dois outros numeros aleatorios entre esses dois maximos, ou seja
    ira gerar um pixel aleatorio dentro dos limites da imagem
    :param max_lin: numero maxixmo de line
    :param max_col: numero maximo de coluna
    :return: retorna uma tupla contendo um numero de lin e um de coluna, tuple(lin, col)
    '''
    #discontando 1 pq ele gera o limite superior
    aux = (randint(0, max_lin-1), (randint(0, max_col-1)))
    return tuple(aux)

# ATENÇÃO: a matrix dada como param é alterada
def generate_noise(matrix, percent, noise):
    '''
    dada uma imagem, irá inserir ruido nela (em apenas uma banda), aleatóriamente, obedecendo uma porcentagem.
    :param matrix: a imagem que se deseja adicionar ruido tipo sal
    :param percent: a porcentagem de ruido que se deseja aplicar na imagem (valor inteiro)
    :param noise: variavel que carrega o tipo de ruido, "salt" = branco, "pepper" = preto
    :return: a matrix referente a imagem, porém acrescido de ruido
    '''

    if noise == "salt":
        noise = 255
    elif noise == "pepper":
        noise = 0

    if type(percent) == int:
        percent = percent/100
    elif type(percent) == float:
        pass

    lista_pixel = []
    nLins, nCols, canais = status_img(matrix)
    qntd_pixels = nLins * nCols
    ch = 0

    dict_pixel = {}
    #gerando posições aleatorias até que o limitar percentual seja atingido
    while len(lista_pixel) < int(qntd_pixels*percent):
        pixel = generate_position(nLins, nCols)
        aux = str(pixel) # guaradaremos os pixels como strings só pra consultar de forma mais eficiente
        if dict_pixel.get(aux, 0) == 0:
            #se entraou aqui é pq nao existe no dict
            dict_pixel[aux] = 1 #coloca no dict
            lista_pixel.append(pixel) #coloca na lista

    for pixel in lista_pixel:
        matrix[pixel[0]][pixel[1]][ch] = noise
        #matrix[pixel[0]][pixel[1]][1] = noise
        #matrix[pixel[0]][pixel[1]][2] = noise

    return matrix

def filtro_media(matrix, limiar):
    '''
    aplica o filtro da media nos pixels da imagem, com uma janela 3x3
    :param matrix: imagem a qual deve ser aplicado o filtro
    :param janela: inteiro que dirá a dimensão da janela, obrseva-se que será apenas um unico numero e a janela sera quadrada
    :param limiar: valor do limiar
    :return: copia da imagem com o filtro aplicado
    '''
    nLins, nCols, canais = status_img(matrix)
    ch = 0
    janela = 3 #sempre 3x3
    #não percorremos a coluna 0, nem a ultima coluna
    #nao percorreremos a linha 0 nem a ultima linha

    #os vizinhos serão guardados em um vetor na seguinte ordem: sentido horário a partir da celula superior ao pixel
    #ou seja a ordem, central, norte, nordeste, leste, sudeste e assim por diante. Nessa organização é ficilitada a aplicação de
    #pesos

    for i in range(1, nLins-1):
        for j in range(1, nCols-1):
            pixels = []
            pixels.append((matrix[i][j][ch])*4)         #central
            pixels.append((matrix[i-1][j][ch])*2)       #norte
            pixels.append(matrix[i-1][j+1][ch])     #nordeste
            pixels.append((matrix[i][j+1][ch])*2)       #leste
            pixels.append(matrix[i+1][j+1][ch])     #sudeste
            pixels.append((matrix[i+1][j][ch])*2)      #sul
            pixels.append(matrix[i+1][j-1][ch])     #sudoeste
            pixels.append((matrix[i][j+1][ch])*2)    #oeste
            pixels.append(matrix[i-1][j-1][ch])     #noroeste

            soma = 0
            for pix in pixels:
                soma += pix
            soma = int(soma/16)

            if abs(soma - pixels[0]) > limiar:
                matrix[i][j][ch] = soma
            else:
                #o pixel não é diferente da vizinhança
                pass
    return matrix

def filtro_moda(matrix):
    '''
    essa função aplica o filtro da moda na imagem dada como entrada
    :param matrix: a matriz referente a imagem a ser processada
    :return: a matriz processada pela função
    '''
    nLins, nCols, canais = status_img(matrix)
    ch = 0
    janela = 3  # sempre 3x3
    # não percorremos a coluna 0, nem a ultima coluna
    # nao percorreremos a linha 0 nem a ultima linha

    # os vizinhos serão guardados em um vetor na seguinte ordem: sentido horário a partir da celula superior ao pixel
    # ou seja a ordem, central, norte, nordeste, leste, sudeste e assim por diante

    for i in range(1, nLins-1):
        for j in range(1, nCols-1):
            pixels = []
            pixels.append((matrix[i][j][ch])*4)         #central
            pixels.append((matrix[i-1][j][ch])*2)       #norte
            pixels.append(matrix[i-1][j+1][ch])     #nordeste
            pixels.append((matrix[i][j+1][ch])*2)       #leste
            pixels.append(matrix[i+1][j+1][ch])     #sudeste
            pixels.append((matrix[i+1][j][ch])*2)      #sul
            pixels.append(matrix[i+1][j-1][ch])     #sudoeste
            pixels.append((matrix[i][j+1][ch])*2)    #oeste
            pixels.append(matrix[i-1][j-1][ch])     #noroeste

            aux = pd.Series(pixels)
            moda = aux.mode().to_list()
            moda = moda[0]

            matrix[i][j][ch] = moda
    return matrix


def filtro_mediana():
    pass