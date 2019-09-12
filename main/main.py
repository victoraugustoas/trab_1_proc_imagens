import proc_img as cv3

PATH_IMG = "../imagens/aviao.jpg"
PATH_DATA = "./data"
IMG = cv3.open_img(PATH_IMG)

# calcula os histogramas em X partições, salva os vetores concatenando-os em um arquivo
print(
    "histograma local" + ":",
    cv3.save_vector(PATH_DATA, cv3.histogram_local(IMG, 10), name="local_hist.txt"),
)

HIST_GLOBAL = cv3.histogram_global(IMG)
# salva os vetores do histograma global
cv3.save_vector(PATH_DATA, HIST_GLOBAL[0], name="histograma_global_blue.txt")
cv3.save_vector(PATH_DATA, HIST_GLOBAL[1], name="histograma_global_green.txt")
cv3.save_vector(PATH_DATA, HIST_GLOBAL[2], name="histograma_global_red.txt")

# gera os histogramas globais da imagem
print("histograma global" + ":", cv3.generate_histograms(*HIST_GLOBAL, PATH_DATA))

# altera o brilho da img
print("brilho" + ":", cv3.save_img(PATH_DATA, "brilho.jpg", cv3.brightness(IMG, 120)))

# converte a img para negativo
print("negativo" + ":", cv3.save_img(PATH_DATA, "negativo.jpg", cv3.negative(IMG)))
