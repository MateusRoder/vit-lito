import os
import timm
import torch
import numpy as np
import pandas as pd
import torchvision

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from helper.rock_patches import RocksDataset
from helper.processing import median_filter, exapand_channel

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from torchvision.transforms import Normalize, Compose

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

dy, dx = 224,224
mean_std_gray = (0.5729325, 0.1901675) # 250x250 patches
mean_std_gray = (0.56521124, 0.21744747) # 200x200 patches

patch_size = 150

rock_norm = Normalize(mean=mean_std_gray[0], std=mean_std_gray[1])
imgnet = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

patch_transformations = Compose([#
                           median_filter(),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                           exapand_channel(),
                           imgnet,
                            ])

batch_size = 16 
n_classes = 7 # 5
fine_tune_epochs = 10
repetitions = 1
drop = torch.nn.Dropout1d(p=0.3)

models = np.loadtxt('vit_models.txt', dtype='str')#[8:]
#models = ['davit_base']

if torch.cuda.is_available():
    device = 'cuda:0'
else: device = 'cpu'

for resnet in models:

    score_lr = np.zeros((repetitions, 4))
    cm = np.zeros((repetitions, n_classes, n_classes))
    fp= np.zeros(repetitions)
    tp = np.zeros(repetitions)
    fn = np.zeros(repetitions)
    tn = np.zeros(repetitions)
    
    if not os.path.exists('./results/'+str(patch_size)+'/'+resnet+'/'):
        os.makedirs('./results/'+str(patch_size)+'/'+resnet+'/')
        print("Created dir:", resnet)

    for i in range(repetitions):
        print("SEED", i)
        print("MODEL", resnet)
        torch.manual_seed(i)
        np.random.seed(i)
        
        df = pd.read_excel('/home/cupertino/roder/pos-doc/imagens/patches/'+str(patch_size)+'/path_classes.xlsx').drop(columns=['Unnamed: 0'])
        xtrain, xtest, ytrain, ytest = train_test_split(np.array(df['id']), np.array(df['classe']), test_size=0.15, stratify=df['classe'], random_state=i)

        train = pd.DataFrame(np.concatenate((xtrain.reshape((-1,1)), ytrain.reshape((-1,1))), axis=1), columns=['id', 'classe'])
        test = pd.DataFrame(np.concatenate((xtest.reshape((-1,1)), ytest.reshape((-1,1))), axis=1), columns=['id', 'classe'])
        
        xtrain, xval, ytrain, yval = train_test_split(np.array(train['id']), np.array(train['classe']), 
                                                    test_size=0.15, stratify=train['classe'], random_state=i)
        train = pd.DataFrame(np.concatenate((xtrain.reshape((-1,1)), ytrain.reshape((-1,1))), axis=1), columns=['id', 'classe'])
        val = pd.DataFrame(np.concatenate((xval.reshape((-1,1)), yval.reshape((-1,1))), axis=1), columns=['id', 'classe'])
        
        train = RocksDataset(root_dir='/home/cupertino/roder/pos-doc/imagens/patches/'+str(patch_size)+'/', label=train, transform=patch_transformations)
        val = RocksDataset(root_dir='/home/cupertino/roder/pos-doc/imagens/patches/'+str(patch_size)+'/', label=val, transform=patch_transformations)
        test = RocksDataset(root_dir='/home/cupertino/roder/pos-doc/imagens/patches/'+str(patch_size)+'/', label=test, transform=patch_transformations)

        model = timm.create_model(resnet, pretrained=True)#, num_classes=1000)
        try:
            out_features = model.head.out_features # 1000
        except:
            out_features = 1000
        fc = torch.nn.Linear(out_features, n_classes)
        fc = fc.to(device)
        
        model = model.to(device)
        # Cross-Entropy loss is used for the discriminative fine-tuning
        criterion = torch.nn.CrossEntropyLoss()
        
        # Creating the optimzers
        optimizer = [torch.optim.AdamW(model.parameters(), lr=1e-4)] # 
        #optimizer = [torch.optim.AdamW(model.head.parameters(), lr=1e-4)] #
        #optimizer.append(torch.optim.AdamW(model.head.parameters(), lr=1e-3))
        optimizer.append(torch.optim.AdamW(fc.parameters(), lr=1e-3))

        # Creating training and validation batches
        train_batch = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
        val_batch = DataLoader(val, batch_size=batch_size//2, shuffle=False, num_workers=0)
        test_batch = DataLoader(test, batch_size=batch_size//2, shuffle=False, num_workers=0, drop_last=True)

        # For amount of fine-tuning epochs
        best_acc, best_epoch = 0, 0
        for e in range(fine_tune_epochs):
            print(f"Epoch {e+1}/{fine_tune_epochs}")

            # Resetting metrics
            train_loss, val_acc, test_acc = 0, 0, 0

            # For every possible batch
            for x_batch, y_batch in tqdm(train_batch):
                x_batch = x_batch.squeeze(1)
                # For every possible optimizer
                for opt in optimizer:
                    # Resets the optimizer
                    opt.zero_grad()

                # Checking whether GPU is avaliable and if it should be used
                if torch.cuda.is_available():
                    # Applies the GPU usage to the data and labels
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                
                # Passing the batch down the model
                y = model(x_batch)
                y = torch.nn.functional.relu(y)
                y = fc(drop(y))

                # Calculating loss
                loss = criterion(y, y_batch)
                
                # Propagating the loss to calculate the gradients
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fc.parameters(), max_norm=20)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)

                # For every possible optimizer
                for opt in optimizer:
                    # Performs the gradient update
                    opt.step()

                # Adding current batch loss
                train_loss += loss.item()
            del loss, x_batch, y_batch, y

            with torch.no_grad():
                # Calculate the test accuracy for the model:    
                for x_batch, y_batch in tqdm(val_batch):
                    x_batch = x_batch.squeeze(1)
                    # Checking whether GPU is avaliable and if it should be used
                    if torch.cuda.is_available():
                        # Applies the GPU usage to the data and labels
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()

                    # Passing the batch down the model
                    y = model(x_batch)
                    y = torch.nn.functional.relu(y)
                    y = torch.nn.functional.softmax(fc(y), dim=1)

                    # Calculating predictions
                    _, preds_ = torch.max(y, 1)

                    # Calculating validation set accuracy
                    val_acc += torch.mean((torch.sum(preds_ == y_batch).float()) / x_batch.size(0))
                del x_batch, y, preds_, y_batch

                preds_, y_true_ = [], []
                
                for x_batch, y_batch in tqdm(test_batch):
                    x_batch = x_batch.squeeze(1)
                    if torch.cuda.is_available():
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()

                    y = model(x_batch)
                    y = torch.nn.functional.relu(y)
                    y = torch.nn.functional.softmax(fc(y), dim=1)
                    _, pred = torch.max(y, 1)

                    test_acc += torch.mean((torch.sum(pred == y_batch).float()) / x_batch.size(0))

                    preds_.append(pred.detach().cpu().numpy())
                    y_true_.append(y_batch.detach().cpu().numpy())
                del x_batch, y, y_batch

            print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc/len(val_batch)} | Test Accuracy: {test_acc/len(test_batch)}")
            acc = val_acc/len(val_batch)
            if acc > best_acc:
                best_acc = acc
                preds = preds_
                y_true = y_true_
                best_epoch = e
                print("Computed best")
                torch.save([model, fc], './results/'+str(patch_size)+'/'+resnet+'/'+resnet+'_best_model.pth')
        
        preds = np.array(preds).reshape((-1, 1))
        y_batch = np.array(y_true).reshape((-1, 1))
        
        score_lr[i, 0] = f1_score(y_batch, preds, average='weighted')
        score_lr[i, 1] = precision_score(y_batch, preds, average='weighted')
        score_lr[i, 2] = recall_score(y_batch, preds, average='weighted')
        score_lr[i, 3] = accuracy_score(y_batch, preds)
        
        print(confusion_matrix(y_batch, preds))
        cm[i, :, :] = confusion_matrix(y_batch, preds)

    #cmatrix = pd.DataFrame(cm.mean(axis=0)).to_excel('./results/'+str(patch_size)+'/'+resnet+'/'+resnet+'_confusion_matrix.xlsx')
    final = pd.DataFrame((score_lr[:, 1].reshape((repetitions, 1))))
    final.columns=['precision']
    final['recall'] = score_lr[:, 2].reshape((repetitions,1))
    final['f1'] = score_lr[:, 0].reshape((repetitions,1))
    final['accuracy'] = score_lr[:, 3].reshape((repetitions,1))
    #final.to_excel('./results/'+str(patch_size)+'/'+resnet+'/'+resnet+'.xlsx')
    del model
    
    
