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
        return result


class MinMaxFilter:
    @staticmethod
    def min_filter(img, pool=(3, 3)):
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
        for x in range(_hb, result.shape[0]-_hb):
            for y in range(_vb, result.shape[1]-_vb):
                result[x, y, ...] = np.amin(result[x-_hb:x+_hb, y-_vb:y+_vb, ...])
        return result[_hb:-_hb:, _vb:_vb:, ...]

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
        for x in range(_hb, result.shape[0]-_hb):
            for y in range(_vb, result.shape[1]-_vb):
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
    params = [0.33, 0.66, 0.33, 0.66]
    _labels = [f"f(min) parameter: ", f"f(max) parameter: ", f"g(min) parameter: ", f"g(max) parameter: "]
    print(f"Image dissection parameters")
    for i in range(len(params)):
        try:
            temp = float(input(_labels[i]))
            params[i] = temp if temp != 0 else params[i]
        except ValueError:
            continue
    img = plt.imread(path)
    dissected = ImageDissector.dissect(img, (params[0], params[1]), (params[2], params[3]))
    bins, hist = ImageAnalyser.hist(img, bin_num)
    plt.figure().add_subplot()
    plt.gca().hist(bins[:-1], bins, weights=hist)
    pass


if __name__ == '__main__':
    images = [p for p in pathlib.Path("img").iterdir() if p.suffix in [".jpg", ".jpeg"]]
    for image in images:
        lab(image, 64)
