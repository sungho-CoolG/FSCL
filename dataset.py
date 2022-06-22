import os
import random
import numpy as np

import torch
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0



class UTKLoader(Dataset):
    def __init__(self,split,ta,sa,data_folder,transform):
        self.data_folder=data_folder
        if split==0:
            self.img_list=os.listdir(self.data_folder+'train/')
        elif split==1:
            self.img_list=os.listdir(self.data_folder+'val/')
        else :
            self.img_list=os.listdir(self.data_folder+'test/')

        self.img_list.sort()
        self.transform=transform
        self.att=[]
        self.split=split
        self.ethnicity_list=[]
        self.age_list=[]
        self.gender_list=[]
        self.ta=ta
        self.sa=sa
        self.split=split

        for i in range(len(self.img_list)):
            self.age_list.append(int(self.img_list[i].split('_')[0])<35)
            self.ethnicity_list.append(int(self.img_list[i].split('_')[2]=='0'))
            self.gender_list.append(int(self.img_list[i].split('_')[1]=='0'))
        
        

    def __getitem__(self, index1):

        index2=random.choice(range(len(self.img_list)))
        age=int(self.age_list[index1])
        gender=int(self.gender_list[index1])
        ethnicity=int(self.ethnicity_list[index1])
        ta=0
        sa=0

        if self.split==0:
            img1=Image.open(self.data_folder+'train/'+self.img_list[index1])
            img2=Image.open(self.data_folder+'train/'+self.img_list[index2])
        elif self.split==1:
            img1=Image.open(self.data_folder+'val/'+self.img_list[index1])
            img2=Image.open(self.data_folder+'val/'+self.img_list[index2])
        else:
            img1=Image.open(self.data_folder+'test/'+self.img_list[index1])
            img2=Image.open(self.data_folder+'test/'+self.img_list[index2])

        if self.ta=='gender':
            ta=gender
        elif self.ta=='age':
            ta=age
        elif self.ta=='ethnicity':
            ta=ethnicity
        
        if self.sa=="gender":
            sa=gender
        elif self.sa=="age":
            sa=age
        elif self.sa=="ethnicity":
            sa=ethnicity
    
        return self.transform(img1),ta,sa


    def __len__(self):
        return (len(self.img_list)-1)


class CelebaLoader(Dataset):
    def __init__(self,split,ta,ta2,sa,sa2,data_folder,transform):
        self.data_folder=data_folder
        if split==0:
            self.img_list=os.listdir(self.data_folder+'train')
        elif split==1:
            self.img_list=os.listdir(self.data_folder+'val')
        else :
            self.img_list=os.listdir(self.data_folder+'test')

        self.split=split
        self.img_list.sort()
        self.transform=transform
        self.att=[]
        

        with open(self.data_folder+'list_attr_celeba.csv','r') as f:
            reader=csv.reader(f)
            att_list=list(reader)
        att_list=att_list[1:]

        with open(self.data_folder+'list_eval_partition.csv','r') as f:
            reader=csv.reader(f)
            eval_list=list(reader)
    
        for i,eval_inst in enumerate(eval_list):
            if eval_inst[1]==str(self.split):
                if att_list[i][0]==eval_inst[0]:
                    self.att.append(att_list[i])
                else:
                    pass

        
        self.att=np.array(self.att)
        self.att=(self.att=='1').astype(int)
        self.ta=ta
        self.ta2=ta2
        self.sa=sa
        self.sa2=sa2

    def __getitem__(self, index1):
        
        ta=self.att[index1][int(self.ta)]
        sa=self.att[index1][int(self.sa)]

        if self.ta2!='None':
            ta2=self.att[index1][int(self.ta2)]
            ta=ta+2*ta2

        if self.sa2!='None':
            sa2=self.att[index1][int(self.sa2)]
            sa=sa+2*sa2

        
        index2=random.choice(range(len(self.img_list)))
        if self.split==0:
            img1=Image.open(self.data_folder+'train/'+self.img_list[index1])
            img2=Image.open(self.data_folder+'train/'+self.img_list[index2])
        elif self.split==1:
            img1=Image.open(self.data_folder+'val/'+self.img_list[index1])
            img2=Image.open(self.data_folder+'val/'+self.img_list[index2])
        else:
            img1=Image.open(self.data_folder+'test/'+self.img_list[index1])
            img2=Image.open(self.data_folder+'test/'+self.img_list[index2])
    
     
        return self.transform(img1),ta,sa


    def __len__(self):
        return len(self.att)
