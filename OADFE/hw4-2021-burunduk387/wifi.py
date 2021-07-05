import numpy as np
import json
import argparse

#Своеобразная линия отреза, подобрана методом проб и ошибок
up = 30
down = -30

#Код Баркера
code = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1], dtype="int8")  
code = np.repeat(code, 5)

parser = argparse.ArgumentParser()
parser.add_argument("input",  metavar="FILENAME", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(str(args.input)) as f:
        data = np.array(f.readlines(), dtype="float64")
        #Свертка двух последовательностей
        cnl = np.convolve(data, code[::-1], mode="same")  
        #Магия, чтобы избежать for
        #Просто берем и "делим" массивчик на -1, 0 и 1, потом множественные 
        #вхождения одного и того же склеиваем
        #Нули убираем, -1 заменяем на 0 и декодируем сообщение
        #Частично позаимствовано с семинара
        out = (cnl > up).astype(np.int8)
        out[cnl < down] = -1
        ind = (out[1:] == out[:-1])
        ind1 = np.concatenate([np.asarray([False,]), ind])
        out = out[~ind1]
        out = out[out != 0]
        out[out == -1] = 0
        message = np.packbits(np.array(out)).tobytes().decode("ascii")
        with open("wifi.json", "w") as f_out:
            json.dump({"message": message}, f_out)
