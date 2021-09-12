import pandas as pd

PTVCsv = pd.read_csv('datasets/ToGraph/PTV_AND_SA3_DATA.csv')


# Makes PTV per person columns
PTVCsv['PEOPLE_PER_BUS_STOP'] = PTVCsv['POPULATION'] / PTVCsv['BusCount']
PTVCsv['PEOPLE_PER_TRAIN_STOP'] = PTVCsv['POPULATION'] / PTVCsv['TrainCount']
PTVCsv['PEOPLE_PER_TRAM_STOP'] = PTVCsv['POPULATION'] / PTVCsv['TramCount']

PTVCsv = PTVCsv[['SA3_CODE', 'SA3_NAME', 'POPULATION', 'PEOPLE_PER_BUS_STOP', 'PEOPLE_PER_TRAIN_STOP', 'PEOPLE_PER_TRAM_STOP']]

PTVCsv.to_csv('datasets/ToGraph/PTV_PER_PERSON.csv', index=False)