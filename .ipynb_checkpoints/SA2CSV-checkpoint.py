import pandas as pd

def getName(string):
    string = string.split(' (', 1)[0]
    return string

SA2df = pd.read_csv('datasets/Aurin/SA2-Code.csv', usecols=['sa2_name_2016', ' sa2_maincode_2016'])
location = ['North', 'East', 'South', 'West']

newSA2df = []

for index, row in SA2df.iterrows():
    SA2String = row['sa2_name_2016']
    SA3Code = str(row[' sa2_maincode_2016'])[0:5]
    
    #start always a valid SA2 name, no extra checks except remove ()
    start = SA2String.split(' - ', 1)[0]
    newSA2df.append([getName(start), SA3Code, row[' sa2_maincode_2016']])
    
    if ' - ' in SA2String:
        
        # check end is not N E S W location, then append if not
        end = SA2String.split(' - ', 1)[1]
        if end not in location:
            newSA2df.append([getName(end), SA3Code, row[' sa2_maincode_2016']])

        
        
finaldf = pd.DataFrame(newSA2df, columns=['SA2_NAME', 'SA3_CODE', 'SA2_CODE'])

finaldf.drop_duplicates(subset='SA2_NAME', keep='first', inplace=True)
                        
finaldf.to_csv('datasets/SA2_SA3.csv', index=False)

