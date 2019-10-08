import proc_img as cv3

from pprint import pprint

PATH_IMG = "../imagens/Lenna.png"
PATH_DATA = "./data"
IMG = cv3.open_img(PATH_IMG)
IMG_CINZA = cv3.open_img(PATH_IMG, gray=True)

# # alteração de brilho
# print("brilho:", cv3.save_img(PATH_DATA, "img_brilho.jpg", cv3.brightness(IMG, -100)))

# # imagem negativa
# print("negativa:", cv3.save_img(PATH_DATA, "img_negativa.jpg", cv3.negative(IMG)))

# hist = cv3.histogram_global(IMG)

# cv3.save_vector(PATH_DATA, hist[0], prefix="blue_", name="hist_global.txt")
# cv3.save_vector(PATH_DATA, hist[1], prefix="green_", name="hist_global.txt")
# cv3.save_vector(PATH_DATA, hist[2], prefix="red_", name="hist_global.txt")

# # histograma global
# print(
#     "Hist. global:",
#     cv3.generate_histograms(*cv3.histogram_global(IMG), PATH_DATA, "global_"),
# )

# histograma local
# print(
#     "Hist. local",
#     cv3.save_vector(
#         PATH_DATA,
#         *cv3.histogram_local(IMG, 5, 1),
#         prefix="hist_local_",
#         name="vectors.txt",
#     ),
# )

# # questão 5
# # equalização de histograma
# print(
#     "equalização de histograma:",
#     cv3.save_img(
#         PATH_DATA,
#         "img_hist_equalizada.jpg",
#         cv3.hist_to_img(
#             IMG_CINZA, cv3.equalize_hist(IMG, cv3.histogram_global(IMG_CINZA))
#         ),
#     ),
# )

# # fatiamento
# cv3.save_img(PATH_DATA, "fatiamento.jpg", cv3.fatiamento(IMG_CINZA))

# # equalização linear
# cv3.save_img(PATH_DATA, "equalização_linear.jpg", cv3.linear_enhancement(IMG, 1.50, 20))

cv3.save_img(PATH_DATA, "quantizada.jpg", cv3.quantization_colors(IMG))

# questão 7
print("sobel ...")
IMG_QUANT = cv3.quantization_colors(IMG_CINZA)
cv3.save_img(PATH_DATA, "sobel.jpg", cv3.filtro_sobel(IMG_QUANT))


# questão 8
print("bic")
IMG_QUANT = cv3.quantization_colors(IMG)
pprint(cv3.bic(IMG_QUANT, 32))