import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds


dataFrame = pd.read_csv('./user_record_db.csv', header=0, 
                         names=['personID', 'contentID', 'eventType', 'timestamp'],
                         engine='python')
print(dataFrame.shape)
print(dataFrame.head(10))

transform = {'VIEW':1, 'LIKE':2, 'COMMENT CREATED':3, 'BOOKMARK':4, 'FOLLOW':5}
print(transform['VIEW'])

ts = dataFrame.copy()
ts = ts.sort_values(by=['personID','contentID'],ascending=True)
#ts = ts.sort_values(by=[], inplace=True)
print(ts.tail(5))

with open('user_record_clean_db.csv', 'w') as f:
    f.write('personID, contentID, eventStrength, timestamp\n')
    for index, row in ts.iterrows():
        f.write(str(row['personID']))
        f.write(',')
        f.write(str(row['contentID']))
        f.write(',')
        f.write(str(transform[row['eventType']]))
        f.write(',')
        f.write(str(row['timestamp']))
        f.write('\n')

        