from mixfit import em_double_cluster
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
import json
from scipy import stats
import matplotlib

def invert(raa, obj, decc):
    return (obj - raa.mean()) / np.cos(decc / 180 * np.pi) + raa.mean()

center_coord = SkyCoord('02h21m00s +57d07m42s')
vizier = Vizier(
    columns=['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE'],
    column_filters={'BPmag': '<16', 'pmRA': '!=', 'pmDE': '!='}, # число больше — звёзд больше
    row_limit=10000
)
stars = vizier.query_region(
    center_coord,
    width=1.0 * u.deg,
    height=1.0 * u.deg,
    catalog=['I/350'], # Gaia EDR3
)[0]

ra = stars['RAJ2000']._data   # прямое восхождение, аналог долготы
dec = stars['DEJ2000']._data  # склонение, аналог широты
x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi) + ra.mean()
x2 = dec 
v1 = stars['pmRA']._data
v2 = stars['pmDE']._data


x = np.column_stack((x1, x2, v1, v2)) 

res = em_double_cluster(x, 0.4, 0.5, np.array([np.mean(x[:, 2]) - 0.5, np.mean(x[:, 3]) + 0.5]),\
                                     np.array([np.mean(x[:, 0]) - 0.5, np.mean(x[:, 1]) - 0.5]),\
                                     np.array([np.mean(x[:, 0]) + 0.5, np.mean(x[:, 1]) + 0.5]),\
                                     300, 100, 10, rtol=1e-5, max_iter=200) 
#print(res)

#ПРОБЛЕМА ОЧЕВИДНО С ДАБЛ КЛАСТЕРОМ, ОСТАЛЬНЫЕ ПУНКТЫ В ПРЕДПОЛОЖЕНИИ
#ЧТО ВСЕ РАБОТАЕТ ДО ЭТОГО

#Переводим x1 обратно в ra 
res[3][0] = invert(ra, *res[3])
res[4][0] = invert(ra, *res[4])

#sigma штука представляющая собой модуль вектора, его обратно переводить не надо
#Модуль длины же, а тут преобразование направления

#Запишем файлик
with open('per.json', 'w') as f:
    json.dump({"size_ratio": 1.2, 
               "motion": { "ra": res[2][0],
                           "dec": res[2][1]},
               "clusters": [
                       {"center": {
                           "ra": res[3][0],
                           "dec": res[3][1]}},
                       {"center": {
                           "ra": res[4][0],
                           "dec": res[4][1]}
                       }]},
                  f,
                  indent=4,
                  separators=(',', ': '))
#Распределения
d1 = stats.multivariate_normal.pdf(x, mean=[*res[3], *res[2]], \
                    cov=np.diag([res[6], res[6], res[7], res[7]]))
d2 = stats.multivariate_normal.pdf(x, mean=[*res[4], *res[2]], \
                    cov=np.diag([res[6], res[6], res[7], res[7]]))
d3 = stats.multivariate_normal.pdf(x[:, 2:], mean=None,  \
                    cov=np.diag([res[5], res[5]]))
ress = res[0] * d1 + res[1] * d2 + (1 - res[0] - res[1]) * d3
d1 = np.divide(res[0] * d1, ress, where=d1!=0, out=np.full_like(d1, 0.33))
d2 = np.divide(res[1] * d2, ress, where=d2!=0, out=np.full_like(d2, 0.33))
d3 = np.divide((1 - res[1] - res[0]) * d3, ress, where=d3!=0, \
                       out=np.full_like(d3, 0.34))
        

colormap = matplotlib.cm.viridis
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

#Первый график
g1 = ax[0].scatter(x=ra, y=x2, c=1-d1-d2, cmap=colormap, s=2, vmin=0, vmax=1)
ax[0].scatter(x=res[3][0], y=res[3][1], s=3, color="black")
ax[0].scatter(x=res[4][0], y=res[4][1], s=3, color="blue")
circle1 = plt.Circle((res[3][0], res[3][1]), np.sqrt(res[6]), fill=False, color="black")
circle2 = plt.Circle((res[4][0], res[4][1]), np.sqrt(res[6]), fill=False, color="blue")
ax[0].add_artist(circle1)
ax[0].add_artist(circle2)
ax[0].set_xlabel("ra")
ax[0].set_ylabel("dec")
ax[0].set_title("Scatter plot of star field points")
ax[0].set_aspect(1)
#Второй грфик
g2 = ax[1].scatter(x=v1, y=v2, c=1-d1-d2, cmap=colormap, s=2, vmin=0, vmax=1)
ax[1].scatter(x=res[2][0], y=res[2][1], s=3, color="red")
circle1 = plt.Circle((res[2][0], res[2][1]), np.sqrt(res[7]), fill=False, color="red")
ax[1].add_artist(circle1)
ax[1].set_xlabel(r"$V_{ra}$")
ax[1].set_ylabel(r"$V_{dec}$")
ax[1].set_title("Scatter plot of proper movements")
ax[1].set_aspect(1)
#Делаем шкалу справа
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(g1, cax=cax)
#Делаем шкалу справа
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(g2, cax=cax)
#Сейвим файлик
fig.tight_layout()
plt.savefig("per.png")

