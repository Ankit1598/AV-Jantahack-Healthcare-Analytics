import numpy as np
import pandas as pd

data_path = "./Train/"
fhc = pd.read_csv( data_path + "First_Health_Camp_Attended.csv" )
shc = pd.read_csv( data_path + "Second_Health_Camp_Attended.csv" )
thc = pd.read_csv( data_path + "Third_Health_Camp_Attended.csv" )
print(fhc.shape, shc.shape, thc.shape)

fhc = fhc[['Patient_ID','Health_Camp_ID','Health_Score']]
fhc = fhc.rename(columns={'Health_Score': 'Outcome'})

shc = shc[['Patient_ID','Health_Camp_ID','Health Score']]
fhc = fhc.rename(columns={'Health Score': 'Outcome'})

thc = thc[['Patient_ID','Health_Camp_ID','Number_of_stall_visited']]
thc = thc[thc['Number_of_stall_visited']>0]
thc = thc.rename(columns={'Number_of_stall_visited': 'Outcome'})
print(fhc.shape, shc.shape, thc.shape)

all_camps = pd.concat([fhc, shc, thc])
all_camps['Outcome'] = 1
print(all_camps.shape)

train = pd.read_csv(data_path + "Train.csv")
print(train.shape)

train = train.merge(all_camps, on=['Patient_ID','Health_Camp_ID'], how='left')
train['Outcome'] = train['Outcome'].fillna(0).astype('int')
train.to_csv(data_path+'train_with_outcome.csv', index=False)