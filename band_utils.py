import rasterio as rio
import os

base = "data"
image_folder = "LC08_L2SP_172061_20150101_20200910_02_T1"

def get_base_image():
    image_path = os.path.join(base, image_folder)
    band_1_file_name = f"{image_folder}_SR_B1.tif"
    band_1_file = os.path.join(image_path, band_1_file_name)
    return band_1_file

def get_band_values(lon, lat):
    src = get_base_image()
    with rio.open(src, 'r') as dataset:
        data = dataset.read(1)
        print(dataset.bounds)



if __name__ == "__main__":
    lon = 34.34607664
    lat = -6.968060476
    get_band_values(lon, lat)