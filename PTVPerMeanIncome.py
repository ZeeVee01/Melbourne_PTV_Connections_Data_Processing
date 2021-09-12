import pandas as pd

PTVCsv = pd.read_csv('datasets/ToGraph/PTV_AND_SA3_DATA.csv')


# Makes PTV per person columns
PTVCsv['MEAN_INCOME_/_BUS_STOP'] = PTVCsv['MEAN_INCOME'] / PTVCsv['BusCount']
PTVCsv['MEAN_INCOME_/_TRAIN_STOP'] = PTVCsv['MEAN_INCOME'] / PTVCsv['TrainCount']
PTVCsv['MEAN_INCOME_/_TRAM_STOP'] = PTVCsv['MEAN_INCOME'] / PTVCsv['TramCount']

PTVCsv = PTVCsv[['SA3_CODE', 'SA3_NAME', 'MEAN_INCOME', 'MEAN_INCOME_/_BUS_STOP', 'MEAN_INCOME_/_TRAIN_STOP', 'MEAN_INCOME_/_TRAM_STOP']]

PTVCsv.to_csv('datasets/ToGraph/PTV_PER_MEAN_INCOME.csv', index=False)