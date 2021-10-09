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
            result = np.where(img < f[0], g[0], g[1])
        return result.astype(dtype=np.uint8)


class MinMaxFilter:
    @staticmethod
    def _apply_pooled(arr, call, fill_val, pool=3):
        if type(arr) is not np.ndarray:
            arr = np.array(arr)
        pool = int(pool) - 1
        result = np.full(shape=(arr.shape[0]-pool, *arr.shape[1:]), fill_value=fill_val, dtype=arr.dtype)
        for i in range(pool):
            result = call(result, arr[i:-(pool-i)])
        return result

    @staticmethod
    def min_filter(img, pool=(3, 3)):
        if type(img) is not np.ndarray:
            img = np.array(img)
        _hb = int(pool[0]/2)
        _vb = int(pool[0]/2)
        """result = np.zeros(shape=(img.shape[0]-2*_hb, img.shape[1]-2*_vb, *img.shape[2:]), dtype=np.uint8)
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                result[x, y] = np.amin(img[x:x + pool[0], y:y + pool[1]], axis=(0, 1))"""

        result = np.zeros(shape=(img.shape[0]+2*_hb, img.shape[1]+2*_vb, *img.shape[2:]), dtype=np.uint8)
        result[_hb:-_hb, _vb:-_vb] = img
        result[_hb:-_hb] = MinMaxFilter._apply_pooled(result, np.minimum, np.iinfo(result.dtype).max, pool[0])
        np.transpose(result, axes=(1, 0, 2))
        result[_vb:-_vb] = MinMaxFilter._apply_pooled(result, np.minimum, np.iinfo(result.dtype).max, pool[1])
        np.transpose(result, axes=(1, 0, 2))
        """result = np.zeros(shape=img.shape)
        for x in range(1, img.shape[0]-1):
            for y in range(1, img.shape[1]-1):
                result[x, y] = np.amin(img[x - _hb:x + _hb + 1, y - _vb:y + _vb + 1], axis=(0, 1))"""
        return result[_hb:-_hb, _vb:-_vb]

    @staticmethod
    def max_filter(img, pool=(3, 3)):
        if type(img) is not np.ndarray:
            img = np.array(img)
        _hb = int(pool[0]/2)
        _vb = int(pool[0]/2)

        result = np.zeros(shape=(img.shape[0]+2*_hb, img.shape[1]+2*_vb, *img.shape[2:]), dtype=np.uint8)
        result[_hb:-_hb, _vb:-_vb] = img
        result[_hb:-_hb] = MinMaxFilter._apply_pooled(result, np.maximum, np.iinfo(result.dtype).min, pool[0])
        np.transpose(result, axes=(1, 0, 2))
        result[_vb:-_vb] = MinMaxFilter._apply_pooled(result, np.maximum, np.iinfo(result.dtype).min, pool[1])
        return result[_hb:-_hb, _vb:-_vb]

    @staticmethod
    def minmax_filter(img, pool=(3, 3)):
        result = MinMaxFilter.min_filter(img, pool)
        result = MinMaxFilter.max_filter(result, pool)
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

    pool_params = [3, 3]
    _pool_labels = [f"x-size: ", f"y-size: "]
    print(f"Minmax pool size (int > 0)")
    for i in range(len(pool_params)):
        try:
            temp = int(input(_pool_labels[i]))
            pool_params[i] = temp if temp > 0 else pool_params[i]
        except ValueError:
            continue

    img = plt.imread(path)
    dissected = ImageDissector.dissect(img, (params[0], params[1]), (params[2], params[3]))
    filtered_min = MinMaxFilter.min_filter(img, (pool_params[0], pool_params[1]))
    # filtered_max = MinMaxFilter.max_filter(img)
    filtered_minmax = MinMaxFilter.minmax_filter(img, (pool_params[0], pool_params[1]))
    # bins, hist = ImageAnalyser.hist(img, bin_num)
    # plt.figure().add_subplot()
    # plt.gca().hist(bins[:-1], bins, weights=hist)

    figures = [plt.figure()]
    figures[0].add_subplot().imshow(img)
    figures.append(plt.figure())
    figures[1].add_subplot().imshow(dissected)
    figures.append(plt.figure())
    figures[2].add_subplot().imshow(filtered_min)
    figures.append(plt.figure())
    figures[3].add_subplot().imshow(filtered_minmax)
    plt.show()
    pass


if __name__ == '__main__':
    images = [p for p in pathlib.Path("img").iterdir() if p.suffix in [".jpg", ".jpeg"]]
    for image in images:
        lab(image, 64)
        if input("Quit? [y]/n: ") != "n":
            break
