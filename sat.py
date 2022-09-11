from PIL import Image
import requests
from sat7_pointer import *
import numpy as np
from tqdm import tqdm
from lxml import etree

# Load Landsat 7 band 1, 2, 3 TIF images and create a composite image
# from the three bands.
#
# The composite image is a greyscale image with a red, green and blue
# channel.
#
# The red channel is the red channel of band 3, the green channel is the
# green channel of band 2 and the blue channel is the blue channel of band 1.
#
# The composite image is saved as a PNG file.

band1 = Image.open("LE07_L1TP_180023_20210712_20210807_02_T1_B1.TIF")
band2 = Image.open("LE07_L1TP_180023_20210712_20210807_02_T1_B2.TIF")
band3 = Image.open("LE07_L1TP_180023_20210712_20210807_02_T1_B3.TIF")

composite = Image.merge("RGB", (band3, band2, band1))

# Load corner coordinates of the image.
#
# The coordinates are stored in a MTL text file.

mtl_data = load_metadata("LE07_L1TP_180023_20210712_20210807_02_T1_MTL.txt")

# Fetch coordinates of a city.
#
# The coordinates of a city are fetched from the OpenStreetMap API.
#
# The coordinates are used to calculate the corner coordinates of the
# composite image.
#
# # City is Belgorod, Russia.

url = "http://nominatim.openstreetmap.org/search?q=Bryansk,+Russia&format=json"
response = requests.get(url)
data = response.json()

# Convert bounding box coordinates to image coordinates.

x0, y1 = lat_lot_to_pixel(
    data[0]["boundingbox"][0], data[0]["boundingbox"][2], mtl_data
)
x1, y0 = lat_lot_to_pixel(
    data[0]["boundingbox"][1], data[0]["boundingbox"][3], mtl_data
)

print(x0, y0)
print(x1, y1)
print(composite.size)
# Crop the image to the bounding box coordinates.

cropped = composite.crop((x0, y0, x1, y1))

cropped.save("cropped.png")

###############################################################################
# LAB 2
###############################################################################

# Load landsat band 4 TIF image.
# Band 4 is near infrared.

band4 = Image.open("LE07_L1TP_180023_20210712_20210807_02_T1_B4.TIF")

# Claculate the NDVI.
#
# The NDVI is calculated by dividing
# difference of the red and near infrared band 4 by the sum of the
# near infrared and near infrared band 4.
#
# The NDVI is calculated for each pixel in the cropped image.
#
# The NDVI is saved as a PNG image.

ndvi = band4.copy().crop((x0, y0, x1, y1))
red = band3.copy().crop((x0, y0, x1, y1))
result = cropped.copy()

print(red.getpixel((0, 0)))
print(ndvi.getpixel((0, 0)))


# Colors for the NDVI.
def get_color(value):
    if value < 0.033:
        return (255, 255, 255)
    elif value < 0.066:
        return (196, 184, 168)
    elif value < 0.1:
        return (180, 150, 108)
    elif value < 0.133:
        return (164, 130, 76)
    elif value < 0.166:
        return (148, 114, 60)
    elif value < 0.2:
        return (124, 158, 44)
    elif value < 0.25:
        return (148, 182, 20)
    elif value < 0.3:
        return (116, 170, 4)
    elif value < 0.35:
        return (100, 162, 4)
    elif value < 0.4:
        return (84, 150, 4)
    elif value < 0.45:
        return (60, 134, 4)
    elif value < 0.5:
        return (28, 114, 4)
    elif value < 0.6:
        return (4, 96, 4)
    elif value < 0.7:
        return (4, 74, 4)
    elif value < 0.8:
        return (4, 56, 4)
    elif value < 0.9:
        return (4, 40, 4)
    else:
        return (0, 0, 0)


for x in tqdm(range(ndvi.size[0]), desc="NDVI"):
    for y in range(ndvi.size[1]):
        r = red.getpixel((x, y))
        nir = ndvi.getpixel((x, y))
        if nir + r == 0:
            result.putpixel((x, y), get_color(0))
        else:
            result.putpixel((x, y), get_color((nir - r) / (nir + r)))

result.save("ndvi.png")

# Calculate FAPAR (Fraction of Absorbed Photosynthetically Active Radiation)
# for each pixel in the cropped image.
#
# Bands 1, 3, 4 are used.

solar_zenith_angle = np.radians(float(mtl_data["SUN_ELEVATION"]))
sensor_zenith_angle = np.radians(0)
sun_sensor_relative_azimuth = np.radians(float(mtl_data["SUN_AZIMUTH"]))

gain = [float(mtl_data["RADIANCE_MULT_BAND_" + str(i)]) for i in [1, 3, 4]]
offset = [float(mtl_data["RADIANCE_ADD_BAND_" + str(i)]) for i in [1, 3, 4]]

dsol = float(mtl_data["EARTH_SUN_DISTANCE"])

pic = [0.643, 0.80760, 0.89472]
k = [0.76611, 0.63931, 0.81037]
theta = [-0.10055, -0.06156, -0.03924]

k = [0.63931, 0.81037, 0.76611]
pic = [0.80760, 0.89472, 0.643]
theta = [-0.06156, -0.03924, -0.10055]

E0 = [1969, 1551, 1044]

cosg = np.cos(solar_zenith_angle) * np.cos(sensor_zenith_angle) + np.sin(
    solar_zenith_angle
) * np.sin(sensor_zenith_angle) * np.cos(sun_sensor_relative_azimuth)
G = (
            np.tan(solar_zenith_angle) ** 2
            + np.tan(sensor_zenith_angle) ** 2
            - 2
            * np.tan(solar_zenith_angle)
            * np.tan(sensor_zenith_angle)
            * np.cos(sun_sensor_relative_azimuth)
    ) ** 0.5

polynoms = np.array(
    [
        [0.27505, 0.35511, -0.004, -0.322, 0.299, -0.0131, 0, 0, 0, 0, 0],
        [-10.036, -0.019804, 0.55438, 0.14108, 12.494, 0, 0, 0, 0, 0, 1],
        [
            0.42720,
            0.069884,
            -0.33771,
            0.24690,
            -1.0821,
            -0.30401,
            -1.1024,
            -1.2596,
            -0.31949,
            -1.4864,
            0,
        ],
    ]
)

blue = band1.copy().crop((x0, y0, x1, y1))
red = band3.copy().crop((x0, y0, x1, y1))
nir = band4.copy().crop((x0, y0, x1, y1))
result = cropped.copy()

f1 = [
    ((np.cos(solar_zenith_angle) * np.cos(sensor_zenith_angle)) ** (k[i] - 1))
    / (np.cos(solar_zenith_angle) + np.cos(sensor_zenith_angle)) ** (1 - k[i])
    for i in range(3)
]
f2 = [
    (1 - theta[i] ** 2) / (1 + 2 * theta[i] * cosg + theta[i] ** 2) ** (3 / 2)
    for i in range(3)
]
f3 = [1 + (1 - pic[i]) / (1 + G) for i in range(3)]
F = [f1[i] * f2[i] * f3[i] for i in range(3)]


def get_color_fapar(value, rho):
    if (0 < rho[0] and rho[0] < 0.257752) \
            and (0 < rho[1] and rho[1] < 0.48407) \
            and (0 < rho[2] and rho[2] < 0.683928) \
            and (rho[0] <= rho[2]) \
            and (rho[2] >= 1.26826 * rho[1]):
        return get_color(value)
    if (rho[0] <= 0) or (rho[1] <= 0) or (rho[2] <= 0):
        return (0, 0, 0)
    if (rho[0] >= 0.257752) or (rho[1] >= 0.48407) or (rho[2] >= 0.683928):
        return (255, 255, 255)
    if (0 < rho[0] and rho[0] < 0.257752) \
            and (0 < rho[1] and rho[1] < 0.48407) \
            and (0 < rho[2] and rho[2] < 0.683928) \
            and (rho[0] > rho[2]):
        return (0, 0, 255)
    if (0 < rho[0] and rho[0] < 0.257752) \
            and (0 < rho[1] and rho[1] < 0.48407) \
            and (0 < rho[2] and rho[2] < 0.683928) \
            and (rho[0] <= rho[2]) \
            and (1.25 * rho[1] > rho[2]):
        return (255, 150, 150)
    if (rho[1] < 0) or (rho[2] < 0):
        return (0, 0, 0)
    if value < 0 or value > 1:
        return (0, 0, 0)

    return (int(180.0 * (1 - value)), 255, 255)


for x in tqdm(range(result.size[0]), desc="FAPAR"):
    for y in range(result.size[1]):
        bands = [blue.getpixel((x, y)), red.getpixel((x, y)), nir.getpixel((x, y))]

        rho_i = [
            (
                    (np.pi * (gain[i] * bands[i] + offset[i]) * dsol ** 2)
                    / (E0[i] * np.cos(sensor_zenith_angle))
            )
            / F[i]
            for i in range(3)
        ]

        g1 = (
                     (polynoms[1, 0] * (rho_i[0] + polynoms[1, 1]) ** 2)
                     + (polynoms[1, 2] * (rho_i[1] + polynoms[1, 3]) ** 2)
                     + polynoms[1, 4] * rho_i[0] * rho_i[1]
             ) / (
                     polynoms[1, 5] * (rho_i[0] + polynoms[1, 6]) ** 2
                     + polynoms[1, 7] * (rho_i[1] + polynoms[1, 8]) ** 2
                     + polynoms[1, 9] * rho_i[0] * rho_i[1]
                     + polynoms[1, 10]
             )
        g2 = (
                     (polynoms[2, 0] * (rho_i[0] + polynoms[2, 1]) ** 2)
                     + (polynoms[2, 2] * (rho_i[2] + polynoms[2, 3]) ** 2)
                     + polynoms[2, 4] * rho_i[0] * rho_i[2]
             ) / (
                     polynoms[2, 5] * (rho_i[0] + polynoms[2, 6]) ** 2
                     + polynoms[2, 7] * (rho_i[2] + polynoms[2, 8]) ** 2
                     + polynoms[2, 9] * rho_i[0] * rho_i[2]
                     + polynoms[2, 10]
             )
        FAPAR = ((polynoms[0, 0] * g2) - polynoms[0, 1] * g1 - polynoms[0, 2]) / (
                (polynoms[0, 3] - g1) ** 2 + (polynoms[0, 4] - g2) ** 2 + polynoms[0, 5]
        )

        result.putpixel((x, y), get_color_fapar(FAPAR, rho_i))

result.save('fapar.png')

root = etree.Element("kml")
doc = etree.SubElement(root, "Document")

start_x = 200
start_y = 300
width = 100
height = 100
result = result.crop((start_x, start_y, start_x + width, start_y + height))

for x in tqdm(range(result.size[0]), desc="KML"):
    for y in range(result.size[1]):
        color = result.getpixel((x, y))
        if color == (0, 0, 0):
            continue
        coord = pixel_to_lat_lot(x + x0 + start_x, y + y0 + start_y, mtl_data)
        neighbour_r = pixel_to_lat_lot(x + 1 + x0 + start_x, y + y0 + start_y, mtl_data)
        neighbour_d = pixel_to_lat_lot(x + x0 + start_x, y + 1 + y0 + start_y, mtl_data)
        neighbour_rd = pixel_to_lat_lot(x + 1 + x0 + start_x, y + 1 + y0 + start_y, mtl_data)
        placemark = etree.SubElement(doc, "Placemark")
        etree.SubElement(placemark, "name").text = f"{x}_{y}"
        etree.SubElement(placemark, "styleUrl").text = f"#style_{x}_{y}"
        polygon = etree.SubElement(placemark, "Polygon")
        outer = etree.SubElement(polygon, "outerBoundaryIs")
        linearring = etree.SubElement(outer, "LinearRing")
        coordinates = etree.SubElement(linearring, "coordinates")
        coordinates.text = f"{coord[0]},{coord[1]},0 {neighbour_r[0]},{neighbour_r[1]},0 {neighbour_rd[0]},{neighbour_rd[1]},0 {neighbour_d[0]},{neighbour_d[1]},0 {coord[0]},{coord[1]},0"
        color = result.getpixel((x, y))
        color = (color[2], color[1], color[0])
        color = '%02x%02x%02x' % color
        color = "64" + color
        style = etree.SubElement(doc, "Style")
        style.set("id", f"style_{x}_{y}")
        polystyle = etree.SubElement(style, "PolyStyle")
        etree.SubElement(polystyle, "color").text = color

with open("fapar.kml", "wb") as f:
    f.write(etree.tostring(root, pretty_print=True))