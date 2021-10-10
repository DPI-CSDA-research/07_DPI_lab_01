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
        _vb = int(pool[1]/2)

        result = np.zeros(shape=(img.shape[0]+2*_hb, img.shape[1]+2*_vb, *img.shape[2:]), dtype=np.uint8)
        result[_hb:-_hb, _vb:-_vb] = img
        result[_hb:-_hb] = MinMaxFilter._apply_pooled(result, np.minimum, np.iinfo(result.dtype).max, pool[0])
        np.transpose(result, axes=(1, 0, 2))
        result[_vb:-_vb] = MinMaxFilter._apply_pooled(result, np.minimum, np.iinfo(result.dtype).max, pool[1])
        np.transpose(result, axes=(1, 0, 2))
        return result[_hb:-_hb, _vb:-_vb]

    @staticmethod
    def max_filter(img, pool=(3, 3)):
        if type(img) is not np.ndarray:
            img = np.array(img)
        _hb = int(pool[0]/2)
        _vb = int(pool[1]/2)

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
    def hist(x, n_bins=20, dynamic_range=True, low=0, high=255):
        n_bins = int(n_bins)
        if type(x) != np.ndarray:
            x = np.array(x)
        if dynamic_range:
            x_min = np.amin(x)
            x_range = np.amax(x) - x_min
        else:
            x_min = low
            x_range = high - x_min
        bin_step = float(x_range) / n_bins
        bins = [bin_step * i for i in range(n_bins + 1)]
        if bin_step > 0:
            result = [0] * n_bins
            x = (x - x_min) / bin_step - 1
            i_x = x.astype(int)
            for item in i_x.flat:
                result[item] += 1
            """for item in x.flat:
                result[int((item - x_min) / bin_step)-1] += 1"""
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
    filtered_max = MinMaxFilter.max_filter(img, (pool_params[0], pool_params[1]))
    filtered_minmax = MinMaxFilter.minmax_filter(img, (pool_params[0], pool_params[1]))

    plot_content = [
        [img],
        [dissected],
        [
            [filtered_min],
            [filtered_max]
        ],
        [filtered_minmax]
    ]
    figures = []

    def build_histograms(collection: list):
        for _item in collection:
            if type(_item) is tuple:
                continue
            if type(_item) is list:
                build_histograms(_item)
            else:
                if len(_item.shape) == 3:
                    for _i in range(_item.shape[2]):
                        collection.append(ImageAnalyser.hist(_item[..., _i], bin_num, False))
                else:
                    collection.append(ImageAnalyser.hist(_item, bin_num, False))

    for item in plot_content:
        if type(item) is list:
            build_histograms(item)

    for item in plot_content:
        fig = plt.figure()
        if type(item) is list and len(item) > 0:
            if type(item[0]) is list:
                axes = fig.subplots(len(item), len(item[0]))
                for i in range(len(item)):
                    if type(item[i]) is list:
                        axes[i][0].imshow(item[i][0])
                        axes[i][0].set_axis_off()
                        for j in range(1, len(item[i])):
                            if type(item[i][j]) is tuple:
                                axes[i][j].hist(item[i][j][0][:-1], item[i][j][0], weights=item[i][j][1])
            else:
                axes = fig.subplots(1, len(item))
                axes[0].imshow(item[0])
                axes[0].set_axis_off()
                for j in range(1, len(item)):
                    if type(item[j]) is tuple:
                        axes[j].hist(item[j][0][:-1], item[j][0], weights=item[j][1])
        figures.append(fig)
    plt.show()


if __name__ == '__main__':
    images = [p for p in pathlib.Path("img").iterdir() if p.suffix in [".jpg", ".jpeg"]]
    for image in images:
        lab(image, 64)
        if input("Quit? [y]/n: ") != "n":
            break
