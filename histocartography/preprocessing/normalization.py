"""Normalization module."""
import logging
import sys
import numpy as np
from PIL import Image


# setup logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::preprocessing::Normalization')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)

def local_normalization(x):

    return x

def staining_normalization(image, method='default'):
    """Staining normalization using default method

    Parameters
    ----------
    image : Numpy Array
        Numpy Array to normalize
    method: str
        Method for normalization. Currently only default
    Returns
    -------
    Numpy Array
        Stain-normalized Numpy Array

    TODO Add documentation on the specifics of this method. A link to a paper or similar.
    """
    
    

    log.info('Input Image size is {}'.format(image.shape))

    Io = 240
    beta = 0.15
    alpha = 1
    maxCRef = np.array([1.9705, 1.0308])
    HERef = np.column_stack(([0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]))
    HERef = HERef.astype(np.float32)
    image = image.astype(np.float32)
    h = image.shape[0]
    w = image.shape[1]
    image = image.reshape([w * h, 3])
    ''' TODO: Fix this warning
        PendingDeprecationWarning: the matrix subclass is not the recommended way to represent 
        matrices or deal with linear algebra 
        (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). 
        Please adjust your code to use regular ndarray.
    '''
    image = np.matrix(image)
    OD = -np.log((image + 1) / Io).astype(np.float32)
    ValidIds = np.where(np.logical_or(np.logical_or(OD[:, 0] < beta, OD[:, 1] < beta), OD[:, 2] < beta) == False)[0]
    ODhat = OD[ValidIds, :]
    D, V = np.linalg.eigh(np.cov(np.transpose(ODhat)))
    ids = sorted(range(len(D)), key=lambda k: D[k])
    D = D[ids[1:]]
    V = V[:, ids[1:]]
    # Checking for completely white images
    if np.sum(np.abs(D)) > 1E-6:
        That = np.dot(ODhat, V)
        Phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(Phi, alpha)
        maxPhi = np.percentile(Phi, 100 - alpha)
        vMin = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        if vMin[0] > vMax[0]:
            HE = np.column_stack((vMin, vMax))
        else:
            HE = np.column_stack((vMax, vMin))
        HE = HE.astype(np.float32)
        OD = OD.astype(np.float32)
        ''' TODO: Fix this warning
        PendingDeprecationWarning: the matrix subclass is not the recommended way to represent 
        matrices or deal with linear algebra 
        (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). 
        Please adjust your code to use regular ndarray.
        '''
        C = np.matrix(np.linalg.lstsq(HE, OD.T, rcond=-1)[0]).T
        maxC = np.percentile(C, 99, 0)
        C[:, 0] = C[:, 0] * maxCRef[0] / maxC[0]
        C[:, 1] = C[:, 1] * maxCRef[1] / maxC[1]
        inorm = (Io * np.exp(-np.dot(HERef, C.T))).T
        inorm[inorm > 255] = 255
        inorm = np.array(inorm).reshape(h, w, 3).astype(np.uint8)
    #endif
        log.info('Normalized Image size is {}'.format(inorm.shape))

    return inorm
    

def get_mask(image, method='default'):
    """Extracts a mask of the non-white region of the image

    Parameters
    ----------
    image : Numpy Array
        Numpy Array to extract the mask from
    method: str
        Method for Mask extraction. Currently only default
    Returns
    -------
    Numpy Array
        Mask Numpy Array
    """
    log.info('Input Image size is {}'.format(image.shape))

    return np.ones_like(image)