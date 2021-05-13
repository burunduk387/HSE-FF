from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.ndimage import rotate
from itertools import repeat
from PIL import Image #Это для ресайза картинки
from photutils.detection import DAOStarFinder

np.random.seed(123) #Чтобы результаты воспроизводились

#!!! КОНТРАСТ ПОЧТИ ВЕЗДЕ ПОДОБРАН РУКАМИ !!!
def save(img, s, vmin=0, vmax=0):
    #Сейвим картинку, можно указать контраст, можно заставить matplotlib разбираться
    if vmax != 0 and vmin != 0:
        plt.imsave(s, Image.fromarray(img).resize((512, 512), Image.ANTIALIAS), vmin=vmin,\
                                                                   vmax=vmax, cmap='gray')
    else:
        plt.imsave(s, Image.fromarray(img).resize((512, 512), Image.ANTIALIAS), cmap='gray')

def create_circular_mask(h, w, radius):
    #Просто делается кружочек из единичек, самым простым методом
    center = (int(w/2), int(h/2))
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


data = fits.open("speckledata.fits")[2].data

#Сохраним усреднённую картинку:
save(np.mean(data, axis=0, dtype=np.float64), "mean.png")

#Теперь возьмём ПРЯМОЕ фурье, сместим нулевую частоту в центр, потом возьмём модуль,
#возведём его в квадрат и усредним
fourier_mean = np.mean(np.abs(np.fft.fftshift(np.fft.fft2(data, s=None,\
                                            axes=(-2, -1)))) ** 2, axis=0)

#Сохраняем картинку
save(fourier_mean, "fourier.png", 7e6, 25e8)

#Я не нашёл встроенного метода, поэтому использую такой способ
mask = create_circular_mask(*fourier_mean.shape, radius=50)

#Вычитаем шумовую подложку. У masked_array True означает Invalid, поэтому 
#отрицание перед маской не надо (маска представляет собой отрисованный кружочек)
#См. https://numpy.org/doc/stable/reference/maskedarray.generic.html
fourier_mean -= np.mean(np.ma.masked_array(fourier_mean, mask))

#Усредняем по углам
rotaver_mean = np.mean(list(map(lambda x: rotate(x, np.random.randint(-180, 180),\
                                        reshape=False), repeat(fourier_mean, 100))), axis=0)
#Сохраняем картинку
save(rotaver_mean, "rotaver.png", 10, 5e9)

#Тут иногда возникает деление на 0 (в силу усреднения по углам такое бывает, да)
#Numpy адекватно с ним справляется, но предпочитает поругаться, пусть не делает так
with np.errstate(divide="ignore", invalid="ignore"):
    #Аккуратно делим попиксельно, зануляем лишнее (тут уже надо отрицание маски), берем
    #обратное фурье и не забываем вернуть обратно при помощи ifftshift
    binary = np.fft.ifftshift(np.fft.ifft2(np.ma.masked_array(np.divide(fourier_mean,\
                            rotaver_mean), ~mask).filled(0), s=None, axes=(-2, -1)))
    #Тут есть комплексные величины, а их непонятно как сохранять 
    #4-х мерную картинку рисовать? :)
    binary = np.abs(binary)
    save(binary, "binary.png", 1e-4, 0.12)

#!!! БОНУС !!!
#fwhm из примера https://photutils.readthedocs.io/en/stable/getting_started.html
#threshold подобран так, чтобы оставалось только 3 пика
daofind = DAOStarFinder(fwhm=4, threshold=0.05) 
sources = daofind(binary)
#Самый мощный пик нас и интересует. Сместим его на дно таблички
sources.sort("peak")
#Евклидово расстояние в пикселях * масштаб
#np.mean на всякий случай, расстояние в теории равно, но если нет - пусть выдает среднее 
dist = np.round(np.mean([np.sqrt((sources["xcentroid"][0] - sources["xcentroid"][2])**2\
      + (sources["ycentroid"][0] - sources["ycentroid"][2])**2) * 0.0206,\
                        np.sqrt((sources["xcentroid"][1] - sources["xcentroid"][2])**2\
      + (sources["ycentroid"][1] - sources["ycentroid"][2])**2) * 0.0206]), 3)
with open("binary.json", "w") as f_out:
            json.dump({"distance": dist}, f_out)

