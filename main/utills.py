import csv
from pprint import pprint
from datetime import datetime as dt

# %H:%M:%S


def csv_to_dict(path):
    """
        Faz a leitura do csv e retorna uma lista com dicts

        return lista com dicts extraídos do csv
    """
    lst = []
    with open(path, "r") as arq:
        reader = csv.DictReader(arq)
        for row in reader:
            lst.append(dict(row))

    return lst

