import numpy as np
from parallelization import parallel_map

def crop_brain(dataobj):
    return dataobj[10:-11,13:-8,:-15]

def gmv_normalize_brain(dataobj):
    return dataobj / np.mean(dataobj)

def downsample_brain(dataobj, factor):

    def downsample_along_axis(array, axis):
        l = array.shape[axis]
        padded_size = int(np.ceil(l / float(factor)))*factor
        begin = (padded_size - l) // 2
        padded_shape = list(array.shape)
        padded_shape[axis] = padded_size
        padded = np.zeros(padded_shape)

        slc = [slice(None)]*array.ndim
        slc[axis] = slice(begin, begin+l)
        padded[slc] = array

        reshaped = list(padded_shape)
        reshaped[axis] = factor
        reshaped.insert(axis, padded_size // factor)

        return np.mean(padded.reshape(reshaped), axis=axis+1)

    for ax in range(dataobj.ndim):
        dataobj = downsample_along_axis(dataobj, ax)

    return dataobj

def compute_mean_brain(data, transform=lambda x:x):
    mean_brain = None

    for image in data.Image:
        cur = transform(image.dataobj)
        if mean_brain is None:
            mean_brain = cur + np.zeros(cur.shape)
        else:
            mean_brain = mean_brain + cur

    return mean_brain / len(data)

def compute_var_brain(data, mean_brain, transform=lambda x:x):
    var_brain = np.zeros(mean_brain.shape)

    for image in data.Image:
        var_brain = var_brain + (transform(image.dataobj) - mean_brain)**2

    var_brain = var_brain / (len(data)-1)
    return var_brain

def compute_t_test(data, condition, **kwargs):

    group0 = data.loc[~condition]
    group1 = data.loc[condition]

    #mean0 = compute_mean_brain(group0, **kwargs)
    #mean1 = compute_mean_brain(group1, **kwargs)

    mean0, mean1 = parallel_map(\
        lambda group: compute_mean_brain(group, **kwargs),
        [group0, group1], n_workers=2)

    #var0 = compute_var_brain(group0, mean0, **kwargs)
    #var1 = compute_var_brain(group1, mean1, **kwargs)

    var0, var1 = parallel_map(\
        lambda pair: compute_var_brain(pair[0], pair[1], **kwargs),
        [(group0, mean0), (group1, mean1)], n_workers=2)

    denom = np.sqrt(var0/len(group0) + var1/len(group1))
    # if variance is zero, set t = 0
    denom[denom == 0.0] = 1.0
    return (mean1 - mean0) / denom
