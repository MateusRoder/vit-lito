
import os
import glob
import pandas as pd
import numpy as np

from PIL import Image

def create_df(path='./imagens/', dy=500, dx=500, Type='*.tif', cnh=1, split=0.75):
    
    root = '/home/cupertino/roder/pos-doc'
    classe_doc = ['amostras_a.xlsx', 'amostras_b.xlsx']
    
    path = root
    pathA = path + str("/imagens/poco-a/")
    pathB = path + str("/imagens/poco-b/")

    path = pathA, pathB
    data = []
    cont = 0
    label = []

    for i, pth in enumerate(path):
        print(pth)
        os.chdir(pth)
        files = glob.glob(Type)
        name = classe_doc[i]
        
        for j in range(len(files)):
            if cnh==1:
                img = Image.open(files[j]).convert('L')
            else:
                img = Image.open(files[j]).convert('RGB')
            
            img = img.resize((dy, dx))
            img = (np.array(img)).reshape((cnh, dy, dx))
            
            data.append(img.astype(np.float64))
            c = pd.read_excel(pth+name).iloc[j,:]
            try:
                label.append(c['Classe'].item().round())
            except:
                label.append(c['Classe'])
            cont+=1
    
    df = (np.reshape(data, ((cont, cnh, dy, dx))))
    lb = (np.reshape(np.array(label), ((cont,1))))
    del data
    perc = int((1-split)*len(df))
    
    xtrain = df[perc:]
    ytrain = lb[perc:]
    xtest = df[:perc]
    ytest = lb[:perc]
    
    os.chdir(root)
    print(xtrain.shape)
    print("Dataset Loaded!", xtrain.min(), xtrain.max())
    
    return xtrain, ytrain, xtest, ytest