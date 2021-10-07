import numpy as np
import matplotlib.pyplot as plt
import pathlib


class ImageDissector:
    @staticmethod
    def dissect(img, f, g):
        if type(img) is not np.ndarray:
            img = np.array(img)
        if abs(f[0]-f[1]) > 0.01:
            factor = (g[1] - g[0]) / (f[1] - f[0])
            result = img * factor
            result[result < g[0]] = g[0]
            result[result > g[1]] = g[1]
        else:
            # mask = np.argwhere(img < f[0])
            """mask = img < f[0]
            img[mask] = g[0]
            img[~mask] = g[1]"""
            result = np.where(img < f[0], g[0], g[1])
        return result.astype(dtype=np.uint8)


class MinMaxFilter:
    @staticmethod
    def min_filter(img, pool=(3, 3)):
        if type(img) is not np.ndarray:
            img = np.array(img)
        _hb = int(pool[0]/2)
        _vb = int(pool[0]/2)
        result = np.zeros(shape=img.shape)
        for x in range(1, img.shape[0]-1):
            for y in range(1, img.shape[1]-1):
                result[x, y] = np.amin(img[x - _hb:x + _hb + 1, y - _vb:y + _vb + 1], axis=(0, 1))
        """_hb = int(pool[0] / 2)
        _vb = int(pool[0] / 2)
        result = np.zeros(shape=img.shape)
        for x in range(-int(result.shape[0]/2), int(result.shape[0]/2)):
            for y in range(-int(result.shape[1]/2), int(result.shape[1]/2)):
                result[x, y] = np.reshape(np.amin(img[x-_hb:x+_hb+1, y-_vb:y+_vb+1], axis=(0, 1)), newshape=(1, 1, 3))"""
        return result

    @staticmethod
    def max_filter(img, pool=(3, 3)):
        if type(img) is not np.ndarray:
            img = np.array(img)
        _hb = int(pool[0]/2)
        _vb = int(pool[0]/2)
        result_shape = (
            img.shape[0] + 2 * _hb,
            img.shape[1] + 2 * _vb,
        )
        result = np.zeros(shape=(result_shape[0], result_shape[1], *img.shape[2:]))
        result[_hb:-_hb:, _vb:_vb:, ...] = img
        for x in range(-result.shape[0]/2, result.shape[0]/2):
            for y in range(-int(result.shape[1]/2), int(result.shape[1]/2)):
                result[x, y, ...] = np.amin(result[x-_hb:x+_hb, y-_vb:y+_vb, ...])
        return result[_hb:-_hb:, _vb:_vb:, ...]

    @staticmethod
    def minmax_filter(img, pool=(3, 3)):
        result = MinMaxFilter.min_filter(img, pool)
        result = MinMaxFilter.min_filter(result, pool)
        return result


class ImageAnalyser:
    @staticmethod
    def hist(x, n_bins=20):
        n_bins = int(n_bins)
        if type(x) != np.ndarray:
            x = np.array(x)
        x_min = np.amin(x)
        x_range = np.amax(x) - x_min
        bin_step = float(x_range) / n_bins
        bins = [bin_step * i for i in range(n_bins + 1)]
        if bin_step > 0:
            result = [0] * n_bins
            for item in x:
                result[int((item - x_min) / bin_step)-1] += 1
        else:
            raise ZeroDivisionError
        return bins, result


def lab(path, bin_num=16):
    params = [75, 200, 75, 200]
    _labels = [f"f(min) parameter: ", f"f(max) parameter: ", f"g(min) parameter: ", f"g(max) parameter: "]
    print(f"Image dissection parameters [0...255]")
    for i in range(len(params)):
        try:
            temp = int(input(_labels[i]))
            params[i] = temp if temp != 0 else params[i]
        except ValueError:
            continue
    img = plt.imread(path)
    dissected = ImageDissector.dissect(img, (params[0], params[1]), (params[2], params[3]))
    filtered_min = MinMaxFilter.min_filter(img)
    # filtered_max = MinMaxFilter.max_filter(img)
    # filtered_minmax = MinMaxFilter.minmax_filter(img)
    # bins, hist = ImageAnalyser.hist(img, bin_num)
    # plt.figure().add_subplot()
    # plt.gca().hist(bins[:-1], bins, weights=hist)

    figures = [plt.figure()]
    figures[0].add_subplot().imshow(img)
    figures.append(plt.figure())
    figures[1].add_subplot().imshow(dissected)
    figures.append(plt.figure())
    figures[2].add_subplot().imshow(filtered_min)
    plt.show()
    pass


if __name__ == '__main__':
    images = [p for p in pathlib.Path("img").iterdir() if p.suffix in [".jpg", ".jpeg"]]
    for image in images:
        lab(image, 64)
