def worldToVoxelCoord (worldCoord, origin, spacing):
    stretchedVoxelCoord= np.absolute (worldCoord - origin)
    stretchedVoxelCoord/ spacing
    return voxelCoord

def normalizePlanes (npzarray):
    maxHU = 400.
    minHU = -1000.

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray [npzarray>1] = 1.
    npzarray [npzarray<0] = 0.
    return npzarray