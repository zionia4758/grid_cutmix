
import pandas as pd
import os
from sklearn.model_selection import train_test_split
def make_csv():
    img_path = "./raw-img/"
    classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    df = pd.DataFrame(columns = ['path','class'])
    
    for c in classes:
        imgs = os.listdir(img_path+c)
        class_df = pd.DataFrame(map(lambda x: [c+'/'+x,c], imgs),columns=['path','class'])
        df = df.append(class_df)
    df.to_csv('./data_csv.csv',index=False)
    print(df)

def split_csv(csv_path = "./data_csv.csv"):
    df = pd.read_csv(csv_path)
    # print(df)
    train_df, test_df = train_test_split(df,stratify=df['class'],train_size=0.8, test_size=0.2)
    # print(train_df)
    print(train_df['class'].value_counts())
    # print(test_df)
    print(test_df['class'].value_counts())
    train_df.to_csv('./train.csv',index=False)
    test_df.to_csv('./test.csv', index=False)    


# make_csv()
split_csv() 
