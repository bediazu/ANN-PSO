import pandas as pd
from pandas.api.types import is_object_dtype

TRAIN_FILE_NAME = 'KDDTrain.txt'
TEST_FILE_NAME = 'KDDTest.txt'


#Transformar cada atributo a formato numerico (1-41)
def transform_to_numeric(df):
    normal_value = -1
    for col in df:
        if is_object_dtype(df[col]):
            #print('Columna objeto a transformar: ',col)
            if col == 41:
                normal_value = transformar_col_to_numeric(df[col],True)
            else:
                transformar_col_to_numeric(df[col])
    return normal_value

def transformar_col_to_numeric(col, isTarget=False):
    array = sorted(col.unique())

    counter = 1
    normal_value = -1
    for item in array:
        if isTarget:
            if item == 'normal':
                normal_value = counter
        col.replace(item, counter, inplace=True)
        counter = counter+1
    return normal_value

def convert_to_bipolar(normal_value, df):
    df[42] = df[41].apply(lambda x: 1 if x==normal_value else -1)

def normalization(df):
    a = 0.1
    b = 0.99
    for col in df:
        if col != 42:
            x_max = df[col].max()
            x_min = df[col].min()

            x = df[col]

            y = (x - x_min)/(x_max - x_min)


            y = (b-a) * y+a
            df[col] = y

df = pd.read_csv('Data/'+TRAIN_FILE_NAME,sep=',', index_col=0, header=None)

normal_value = transform_to_numeric(df)
convert_to_bipolar(normal_value,df)
normalization(df)

df.to_csv('train', index=None, header=None)
print('Data de Training normalizada y guardada con exito')

df = pd.read_csv('Data/'+TEST_FILE_NAME,sep=',', index_col=0, header=None)

normal_value = transform_to_numeric(df)
convert_to_bipolar(normal_value,df)
normalization(df)

df.to_csv('test', index=None, header=None)
print('Data de Testing normalizada y guardada con exito')