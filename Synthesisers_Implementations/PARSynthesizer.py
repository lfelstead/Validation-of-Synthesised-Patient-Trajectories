from sdv.sequential import PARSynthesizer
import pandas as pd
import numpy as np

data = pd.read_csv('labeventssyn.csv') 

data1 = data[['hadm_id', 'charttime', 'label']]
# i think it might be running out of memory when using the whole dataset causing segmentation errors
# loading too much data into memory, insufficient ram 
data1 = data1[:80] # use a smaller dataset while testing
print(data1)

#data1 = getRelativeTime(data, 'charttime', 'admittime', 'label')

#from sdv.metadata import MultiTableMetadata
#metadata = MultiTableMetadata()
from sdv.metadata import SingleTableMetadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data1)

metadata.update_column( column_name='hadm_id', sdtype='id')
metadata.update_column( column_name='charttime', sdtype= "datetime", datetime_format= "%Y-%m-%d %H:%M:%S")
metadata.set_sequence_key( column_name='hadm_id')
metadata.set_sequence_index(column_name='charttime')


print(metadata)

synthesizer = PARSynthesizer(metadata, context_columns=[])

synthesizer.fit(data1)
synthetic_data = synthesizer.sample(num_sequences=20)
print(synthetic_data)