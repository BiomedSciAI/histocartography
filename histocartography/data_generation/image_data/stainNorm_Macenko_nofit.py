import numpy as np

# Normalizes the supplied image


def stainingNorm_Macenko_nofit(image):
    Io = 240
    beta = 0.15
    alpha = 1

    maxCRef = np.array([1.9705, 1.0308])

    HERef = np.column_stack(
        ([0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]))
    HERef = HERef.astype(np.float32)

    image = image.astype(np.float32)

    h = image.shape[0]
    w = image.shape[1]

    image = image.reshape([w * h, 3])
    image = np.matrix(image)
    OD = -np.log((image + 1) / Io).astype(np.float32)

    ValidIds = np.where(np.logical_or(np.logical_or(
        OD[:, 0] < beta, OD[:, 1] < beta), OD[:, 2] < beta) == False)[0]
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
        C = np.matrix(np.linalg.lstsq(HE, OD.T, rcond=-1)[0]).T

        maxC = np.percentile(C, 99, 0)

        C[:, 0] = C[:, 0] * maxCRef[0] / maxC[0]
        C[:, 1] = C[:, 1] * maxCRef[1] / maxC[1]

        inorm = (Io * np.exp(-np.dot(HERef, C.T))).T
        inorm[inorm > 255] = 255

        inorm = np.array(inorm).reshape(h, w, 3).astype(np.uint8)
    # endif

    return inorm
# enddef
