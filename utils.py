

import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.externals import joblib

FID = './data/IdLookupTable.csv'
def submission(y_pred):
    y_pred = y_pred * 48.0 + 48.0
    y_pred = y_pred.clip(0, 96)

    lookup_table = read_csv(FID)
    cols = joblib.load('data/cols.pkl') 
    
    values = []
    for index, row in lookup_table.iterrows(): 
        values.append((
            row['RowId'],
            y_pred[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]], 
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location')) 
    submission.to_csv('data/submission.csv', index=False)
