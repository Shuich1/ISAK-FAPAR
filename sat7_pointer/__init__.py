import os
import re
import csv
import pyproj
import math
import numpy as np

def load_metadata(mtl_file):
    mtl_data = {}
    with open(mtl_file) as f:
        mtl = csv.reader(f, delimiter='=')
        try:
            for row in mtl:
                mtl_data[row[0].strip()] = row[1]
        except IndexError:
            pass
    
    mtl_data['width'] = float(mtl_data['REFLECTIVE_SAMPLES'])
    mtl_data['height'] = float(mtl_data['REFLECTIVE_LINES'])

    mtl_data['lat_lon_corners'] = np.array([
        [float(mtl_data['CORNER_UL_LAT_PRODUCT']), float(mtl_data['CORNER_UL_LON_PRODUCT']), 0],
        [float(mtl_data['CORNER_UR_LAT_PRODUCT']), float(mtl_data['CORNER_UR_LON_PRODUCT']), 0],
        [float(mtl_data['CORNER_LR_LAT_PRODUCT']), float(mtl_data['CORNER_LR_LON_PRODUCT']), 0],
        [float(mtl_data['CORNER_LL_LAT_PRODUCT']), float(mtl_data['CORNER_LL_LON_PRODUCT']), 0]
    ])

    mtl_data['ecef'] = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    mtl_data['lla'] = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    x, y, z = pyproj.transform(mtl_data['lla'], mtl_data['ecef'], 
        mtl_data['lat_lon_corners'][:, 1],
        mtl_data['lat_lon_corners'][:, 0],
        mtl_data['lat_lon_corners'][:, 2],
        radians=False)
    
    mtl_data['ecef_corners'] = np.array([x, y, z]).T

    vector_a = mtl_data['vector_a'] = mtl_data['ecef_corners'][1, :] - mtl_data['ecef_corners'][0, :]
    vector_b = mtl_data['vector_b'] = mtl_data['ecef_corners'][2, :] - mtl_data['ecef_corners'][0, :]
    vector_c = mtl_data['vector_c'] = mtl_data['ecef_corners'][3, :] - mtl_data['ecef_corners'][0, :]

    altitude = mtl_data['altitude'] = 710e3
    vector_s = mtl_data['vector_s'] = (vector_a + vector_c) / 2 - altitude * np.cross(vector_a, vector_c) / np.linalg.norm(np.cross(vector_a, vector_c))

    b = np.array([vector_b - (vector_a + vector_c)])
    A = np.array([vector_a - vector_s, vector_c - vector_s, vector_b - vector_s])
    mnt = np.linalg.solve(A.T, b.T)
    mtl_data['mu'], mtl_data['nu'] = mnt[0][0], mnt[1][0]

    return mtl_data

def pixel_to_lat_lot(x, y, mtl_data):
    k = (x - 0.5) / mtl_data['width']
    r = (y - 0.5) / mtl_data['height']

    y = k*(mtl_data['vector_a'] + mtl_data['mu'] * (mtl_data['vector_a'] - mtl_data['vector_s'])) \
        + r * (mtl_data['vector_c'] + mtl_data['nu'] * (mtl_data['vector_c'] - mtl_data['vector_s']))
    B = np.array([mtl_data['vector_a'], mtl_data['vector_c'], mtl_data['vector_s']-y]).T
    ags = np.linalg.solve(B, y.T)
    al = ags[0]
    gm = ags[1]
    ecef = mtl_data['ecef_corners'][0, :] + al*mtl_data['vector_a'] + gm*mtl_data['vector_c']
    lat, lon, alt = pyproj.transform(mtl_data['ecef'], mtl_data['lla'], ecef[0], ecef[1], ecef[2], radians=False)
    return [lat, lon]

def lat_lot_to_pixel(lat, lon, mtl_data):
    target_vector = np.array([lat, lon, 0])

    x1, y1, z1 = pyproj.transform(mtl_data['lla'], mtl_data['ecef'],
        target_vector[1], target_vector[0], target_vector[2], radians=False)
    x2, y2, z2 = pyproj.transform(mtl_data['lla'], mtl_data['ecef'],
        mtl_data['lat_lon_corners'][0, 1], mtl_data['lat_lon_corners'][0, 0], mtl_data['lat_lon_corners'][0, 2], radians=False)
    
    x = np.array([x1-x2, y1-y2, z1-z2])
    v = np.cross(mtl_data['vector_c'], mtl_data['vector_a'])
    xh = np.divide(np.dot(v.T, x), np.linalg.norm(v))
    x1, y1, z1 = pyproj.transform(mtl_data['lla'], mtl_data['ecef'],
        target_vector[1], target_vector[0], -xh, radians=False)
    x = np.array([x1-x2, y1-y2, z1-z2])
    A = np.array([
        (1+mtl_data['mu'])*mtl_data['vector_a'] - mtl_data['mu']*mtl_data['vector_s'], 
        (1+mtl_data['nu'])*mtl_data['vector_c'] - mtl_data['nu']*mtl_data['vector_s'], 
        x-mtl_data['vector_s']
    ]).T
    z = np.linalg.solve(A, x.T)

    k = round(z[0], 4)
    r = round(z[1], 4)

    target_x = min([max([1, round(k*mtl_data['width'] + 0.5)]), mtl_data['width']])
    target_y = min([max([1, round(r*mtl_data['height'] + 0.5)]), mtl_data['height']])

    return target_x, target_y
