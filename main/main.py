import proc_img as cv3
import threading

path_img = "../../imagens/aviao.jpg"
img = cv3.open_img(path_img)

print("histograma local" + ":", cv3.histogram_local(img, 10, "./data"))

print(
    "histograma global" + ":",
    cv3.generate_histograms(*cv3.histogram_global(img), "./data"),
)


print("brilho" + ":", cv3.save_img("./data/", "brilho.jpg", cv3.brightness(img, 120)))
print("negativo" + ":", cv3.save_img("./data", "negativo.jpg", cv3.negative(img)))
