from tools import *


from pylab import *
import numpy as np
from PIL import Image, ImageDraw


def convolution(input_mass, control_mass):
    '''
    Функция связывает входной сигнал с управляющим
    :param input_mass: массив входного сигнала
    :param control_mass: массив управляющего сингала
    :return: элементы массива свертки
    '''
    N, M = len(input_mass), len(
        control_mass)  # Размер входного и управляющего массивов
    conv_mass = []  # Массив, заполняемый элементами свертки
    sum_of_conv = 0
    for k in range(N + M - 1):
        for m in range(M):
            if k - m < 0:
                pass
            if k - m > N - 1:
                pass
            else:
                sum_of_conv += input_mass[k - m] * control_mass[m]

        conv_mass.append(sum_of_conv)
        sum_of_conv = 0
    return conv_mass

# Фильтр назких частот


def low_pass_filter(m=32, dt=0.001, fc=100):
    '''
    Функция для фильтрации низких частот
    :param m: ширина окна (чем больше m, тем круче переходная характеристика фильтра
    :param dt: шаг дискретизации
    :param lpw: вес фильтра
    :return: отфильтрованный массив
    '''

    # расчитвываем веса прямоугольной функции
    arg = 2 * fc * dt
    lpw = []
    lpw.append(arg)
    arg *= np.pi
    for i in range(1, m + 1):
        lpw.append(np.sin(arg * i) / (np.pi * i))

    # Для трапецивидной
    lpw[m] /= 2
    # Применяем окно Поттера сглаживания окно p310 для сглаживания
    # Это окно требует 4 константы:
    d = [0.35577019, 0.24369830, 0.07211497, 0.00630165]
    summ = lpw[0]
    for i in range(1, m + 1):
        summ2 = d[0]
        arg = (np.pi * i) / m
        for k in range(1, 4):
            summ2 += 2 * d[k] * np.cos(arg * k)
        lpw[i] *= summ2
        summ += 2 * lpw[i]
    # Делаем нормировку
    for i in range(m + 1):
        lpw[i] /= summ

    # for i in range(100):
    #     lpw.append(0)
    # Доделать m+1 в 2m+1
    lpw = lpw[::-1] + lpw[1:]
    # for i in range(len(lpw)):
    #     lpw[i] *= len(lpw)
    return lpw


# Фильтр высоких чатсот
def high_pass_filter(m, dt, fc):
    '''
    Фильтр высоких частот
    :param m: отвечает за крутизну фильтра
    :param dt: шаг дискретизации
    :param fc: частота среза
    :return: характеристику фильтра
    '''
    lpw = low_pass_filter(m, dt, fc)
    hpw = []
    for i in range(2 * m + 1):
        if i == m:
            hpw.append(1 - lpw[i])
        else:
            hpw.append(-lpw[i])
    return hpw


# Полосовой фильтр
def bend_pass_filter(m, dt, fc1, fc2):
    '''
    Полосовой фильтр
    :param m: отвечает за крутизну
    :param dt: шаг дискретизации
    :param fc1: входная частота фильтра
    :param fc2: выходная частота фильтра
    :return: характеристику фильтра
    '''
    lpw1 = low_pass_filter(m, dt, fc1)
    lpw2 = low_pass_filter(m, dt, fc2)
    bpw = []
    for i in range((2 * m) + 1):
        bpw.append(lpw2[i] - lpw1[i])
    return bpw


def image_conv(input_mass, control_mass, w, h, m):
    print(input_mass)
    print(control_mass)
    print([w, h, m])
    '''
        Функция реализует свертку изображения и управляющего сигнала
    :param input_mass: матрица изображения для обработки
    :param control_mass: массив управляющего сигнала
    :param w: ширина изображения
    :param h: высота изображения
    :param m: крутость переходной характеристики управляющего массива
    :return: отфильтрованное изображения
    '''
    data_conv = []
    for i in range(h):
        temp = convolution((input_mass[i]), control_mass)
        data_conv.append(temp[m:(w + m)])

    return data_conv


def normalization(mass: list, dim: int, N=255):
    '''
    Функция нормировки
    :param mass: одномерный список значений оттенков серости пикселей
    :param dim: размерность массива
    :param N: количество элементов
    :return: норма
    '''
    if dim == 1:
        norm = []
        min_mass = min(mass)
        max_mass = max(mass)
        for pix in mass:
            norm.append(int(((pix - min_mass) / (max_mass - min_mass)) * N))

        return norm

    norm_mass = []
    width, height = len(mass[0]), len(mass)
    for row in range(height):
        for col in range(width):
            norm_mass.append(mass[row][col])
    norm_mass = normalization(norm_mass, dim=1)

    return np.array(norm_mass).reshape(height, width)


def drawing_image_new(matrix_pixels, width, height):

    # создаем пустую картинку в оттенках серого с шириной w и высотой h
    image_new = Image.new('L', (width, height))
    draw = ImageDraw.Draw(image_new)  # Запускаем инструмент для рисования

    # нормализуем значения оттенков серого
    image_new_norm = normalization(matrix_pixels, dim=2)

    # заполняем значения пикселей новой картинки оттенками серого входного списка
    i = 0
    for y in range(height):
        for x in range(width):
            draw.point((x, y), int(image_new_norm[y][x]))
            i += 1

    return image_new


def count_of_stones():
    file = 'src.jpg'
    image = Image.open(file).convert('L')
    width, height = image.size[0], image.size[1]
    matrix_pixels = np.array(image).reshape(height, width)

    # Произодим пороговую фильтрацию
    # for row in range(height):
    #     for col in range(width):
    #         if matrix_pixels[row][col] < 150:
    #             matrix_pixels[row][col] = 0
    #         else:
    #             matrix_pixels[row][col] = 255

    bin_image = drawing_image_new(matrix_pixels, width, height)
    bin_image.show()
    bin_image.save('images/size_of_object/bin_stones.jpg')

    print('lpf...')
    # Фмльтрация нижних частот
    m, dt, fc = 32, 1 / width, 100
    print('drawLFD...')
    control_mass = low_pass_filter(m, dt, fc)
    print('drawConv...')
    conv_mass = image_conv(matrix_pixels, control_mass, width, height, m)
    print('draw...')
    conv_image = drawing_image_new(conv_mass, width, height)
    print('drawAfter...')
    conv_image.show()

    print('hpf...')
    dt = 0.001
    # Фильтр высоких частот
    control_mass = high_pass_filter(m, dt, fc)
    conv_mass = image_conv(matrix_pixels, control_mass, width, height, m)
    conv_image = drawing_image_new(conv_mass, width, height)
    conv_image.show()

    print('strings...')
    # Полосовой фильтр
    control_mass = bend_pass_filter(m=32, dt=0.001, fc1=100, fc2=200)
    conv_mass = image_conv(matrix_pixels, control_mass, width, height, m)
    conv_image = drawing_image_new(conv_mass, width, height)
    conv_image.show()


if __name__ == '__main__':
    count_of_stones()
