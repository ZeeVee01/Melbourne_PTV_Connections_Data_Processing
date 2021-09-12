import pandas as pd
import sys

# csv to process, either tram, train or bus
toProcess = pd.read_csv(sys.argv[1], usecols=['STOP_ID', ' STOP_NAME'])

# SA2 csv to combine with
SA2Number = pd.read_csv('datasets/SA2_SA3.csv')

# empty list to store sa2 name info
SA2Array = []

for index, row in toProcess.iterrows():
    
    # gets the SA3 name out of the stop name (SA3NAME)
    stopSA2 = row[' STOP_NAME'].split(' (', 1)[1].split(')', 1)[0]
    SA2Array.append(stopSA2)
    
# adds SA2 Name column to process csv
toProcess['SA2_NAME'] = SA2Array

merged = pd.merge(left=toProcess, right=SA2Number, how='left', left_on='SA2_NAME', right_on='SA2_NAME')

merged = merged[['STOP_ID', ' STOP_NAME', 'SA2_NAME', 'SA3_CODE', 'SA2_CODE']]

merged.to_csv(sys.argv[2], index=False)