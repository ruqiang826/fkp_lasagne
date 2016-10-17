

FID = './data/IdLookupTable.csv'
import pandas as pd
from sklearn.externals import joblib
def submission(y_pred):
    y_pred = y_pred * 48.0 + 48.0

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
