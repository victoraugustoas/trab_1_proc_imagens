import proc_img as cv3

PATH_IMG = "../imagens/aviao.jpg"
PATH_DATA = "./data"
IMG = cv3.open_img(PATH_IMG)
IMG_CINZA = cv3.open_img(PATH_IMG, gray=True)

# questão 6
print("filtros espaciais")
IMG_RUIDO = cv3.generate_noise(IMG, 50, "salt")
cv3.save_img(PATH_DATA, "img_ruido.jpg", IMG_RUIDO)

cv3.save_img(PATH_DATA, "mediana_noise.jpg", cv3.filtro_mediana(IMG_RUIDO, 3))
cv3.save_img(PATH_DATA, "moda_noise.jpg", cv3.filtro_moda(IMG_RUIDO, 3))
cv3.save_img(PATH_DATA, "media_noise.jpg", cv3.filtro_media(IMG_RUIDO, 3))
