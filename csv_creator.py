import os
# import numpy
# import rasterio
import pandas as pd
# import matplotlib.pyplot as plt

soc_csv_file_location = "data/naforma_standtable.csv"
ml_csv_file_location = "data/ml.csv"
ml = open(ml_csv_file_location, "w")
row_string = f"id,soc,lon,lat,top,bottom,bd,sand,silt,clay,phh2o,phkcl\n"
ml.write(row_string)
df = pd.read_csv(soc_csv_file_location)

for id, row in df.iterrows():
    row_string = f"{id},{row['oc']},{row['lon']},{row['lat']}," \
                 f"{row['top']},{row['bottom']},{row['bd']},{row['sand']}," \
                 f"{row['silt']},{row['clay']},{row['phh2o']},{row['phkcl']}\n"
    ml.write(row_string)

ml.close()
print("done")


