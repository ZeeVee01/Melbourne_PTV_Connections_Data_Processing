import pandas as pd

PTVCsv = pd.read_csv('datasets/ToGraph/PTV_AND_SA3_DATA.csv')


# Makes PTV per person columns
PTVCsv['TOTAL_INCOME_/_BUS_STOP'] = PTVCsv['TOTAL_INCOME'] / PTVCsv['BusCount']
PTVCsv['TOTAL_INCOME_/_TRAIN_STOP'] = PTVCsv['TOTAL_INCOME'] / PTVCsv['TrainCount']
PTVCsv['TOTAL_INCOME_/_TRAM_STOP'] = PTVCsv['TOTAL_INCOME'] / PTVCsv['TramCount']

PTVCsv = PTVCsv[['SA3_CODE', 'SA3_NAME', 'TOTAL_INCOME', 'TOTAL_INCOME_/_BUS_STOP', 'TOTAL_INCOME_/_TRAIN_STOP', 'TOTAL_INCOME_/_TRAM_STOP']]

PTVCsv.to_csv('datasets/ToGraph/PTV_PER_TOTAL_INCOME.csv', index=False)