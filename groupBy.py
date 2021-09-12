import pandas as pd
import sys

Income = pd.read_csv('datasets/Aurin/SA3-Personal-Income-2015.csv', usecols=[' sa3_code_2016', ' sa3_name16', ' mean_aud', ' median_aud'])

Population = pd.read_csv('datasets/Aurin/SA3-Population.csv', usecols=['estmtd_rsdnt_ppltn_smmry_sttstcs_30_jne_prsns_ttl_nm', ' population_density_as_at_30_june_population_density_personskm2', ' sa3_code_2016'])
Population.columns = ['POPULATION', 'POP_DENSITY_KM2', 'SA3_CODE']

LandArea = pd.read_csv('datasets/Aurin/SA3-LandArea.csv')

PTVBus = pd.read_csv('datasets/PTV-Bus-SA3.csv', usecols=['SA3_CODE'])
PTVTrain = pd.read_csv('datasets/PTV-Train-SA3.csv', usecols=['SA3_CODE'])
PTVTram = pd.read_csv('datasets/PTV-Tram-SA3.csv', usecols=['SA3_CODE'])

#BusDF = PTVBus.groupby('SA3_CODE').size()
BusDF = PTVBus.groupby('SA3_CODE').SA3_CODE.agg('count').to_frame('BusCount').reset_index()
TrainDF = PTVTrain.groupby('SA3_CODE').SA3_CODE.agg('count').to_frame('TrainCount').reset_index()
TramDF = PTVTram.groupby('SA3_CODE').SA3_CODE.agg('count').to_frame('TramCount').reset_index()

# Merge all
merge1 = pd.merge(left=Income, right = BusDF, how='left', left_on=' sa3_code_2016', right_on='SA3_CODE')
merge2 = pd.merge(left=merge1, right = TrainDF, how='left', left_on=' sa3_code_2016', right_on='SA3_CODE')
merge3 = pd.merge(left=merge2, right = TramDF, how='left', left_on=' sa3_code_2016', right_on='SA3_CODE')
merge4 = pd.merge(left=merge3, right = Population, how='left', left_on=' sa3_code_2016', right_on='SA3_CODE')
finalMerge = pd.merge(left=merge4, right = LandArea, how='left', left_on=' sa3_code_2016', right_on='SA3_CODE_2016')

#Add Total Income column, population * mean income
finalMerge['TOTAL_INCOME'] = finalMerge['POPULATION'] * finalMerge[' mean_aud']

# Fix headings
finalMerge = finalMerge[[' sa3_code_2016', ' sa3_name16', 'POPULATION', 'POP_DENSITY_KM2', 'AREA_ALBERS_SQKM', ' mean_aud', ' median_aud', 'TOTAL_INCOME', 'BusCount', 'TrainCount', 'TramCount']]
finalMerge.columns = ['SA3_CODE', 'SA3_NAME', 'POPULATION', 'POP_DENSITY_KM2', 'LAND_AREA_KM2', 'MEAN_INCOME', 'MEDIAN_INCOME', 'TOTAL_INCOME', 'BusCount', 'TrainCount', 'TramCount']

# To CSV
finalMerge.to_csv(sys.argv[1], index=False)


