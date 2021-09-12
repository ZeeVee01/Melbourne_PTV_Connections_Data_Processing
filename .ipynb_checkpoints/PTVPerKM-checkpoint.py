import pandas as pd

PTVCsv = pd.read_csv('datasets/ToGraph/PTV_AND_SA3_DATA.csv')

# Makes PTV per km2 columns
PTVCsv['BUS_STOP_PER_KM2'] = PTVCsv['BusCount'] / PTVCsv['LAND_AREA_KM2']
PTVCsv['TRAIN_STOP_PER_KM2'] =  PTVCsv['TrainCount']/ PTVCsv['LAND_AREA_KM2']
PTVCsv['TRAM_STOP_PER_KM2'] =  PTVCsv['TramCount'] / PTVCsv['LAND_AREA_KM2']

PTVCsv = PTVCsv[['SA3_CODE', 'SA3_NAME', 'LAND_AREA_KM2', 'BUS_STOP_PER_KM2', 'TRAIN_STOP_PER_KM2', 'TRAM_STOP_PER_KM2']]

PTVCsv.to_csv('datasets/ToGraph/PTV_PER_KM.csv', index=False)