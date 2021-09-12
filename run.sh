#!/bin/bash

python SA2CSV.py

#Preprocessing
python processing.py datasets/Aurin/PTV-Tram-Stops.csv datasets/PTV-Tram-SA3.csv
python processing.py datasets/Aurin/PTV-Train-Stops.csv datasets/PTV-Train-SA3.csv
python processing.py datasets/Aurin/PTV-Bus-Stops.csv datasets/PTV-Bus-SA3.csv

#Run Groupby
python groupBy.py datasets/ToGraph/PTV_AND_SA3_DATA.csv

#Run programs to make better datasets to graph
python PTVPerKM.py
python PTVPerPer.py
python PTVPerMeanIncome.py
python PTVPerTotalIncome.py
