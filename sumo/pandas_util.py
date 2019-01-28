
def nan_rows(df):
    return df[df.isnull().any(axis=1)]

