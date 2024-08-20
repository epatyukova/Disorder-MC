#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:53:59 2023

@author: patyukoe
"""

import numpy as np
import re

    
elem_list=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', \
            'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', \
            'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', \
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', \
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',\
            'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',\
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', \
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',\
            'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg']

def proper_split(se):
    split=[]
    for i,symbol in enumerate(se):
        if(symbol=='(' or symbol==')'or symbol=='[' or symbol==']'):
            split.append(symbol)
        elif(re.search('[A-Z]',symbol)):
            if(i<len(se)-1):
                if(re.search('[a-z]',se[i+1])):
                    split.append(se[i:i+2])
                else:
                    split.append(symbol)
            else:
                split.append(symbol)
        elif(re.search('[0-9]',symbol)):
            split.append(symbol)
        elif(re.search('\.',symbol)):
            split.append(symbol)
    return split


def join_numbers(s):
    join=np.zeros(len(s)-1)
    for i,symbol in enumerate(s[:-1]):
        if(re.search('[0-9]',symbol)):
            if(re.search('[0-9]',s[i+1])):
                join[i]=1
            elif(re.search('\.',s[i+1])):
                join[i]=1
        elif(re.search('\.',symbol)):
            if(re.search('[0-9]',s[i+1])):
                join[i]=1
    left=[]
    right=[]
    for i in range(len(join)-1):
        if(join[i]==0 and join[i+1]==1):
            left.append(i)
        elif(join[i]==1 and join[i+1]==0):
            right.append(i)
    if(len(right)<len(left)):
        right.append(len(s)-1)
    lenth=0
    for i in range(len(left)):
        s[left[i]+1-lenth:right[i]+2-lenth]=[''.join(s[left[i]+1-lenth:right[i]+2-lenth])]      
        lenth+=right[i]-left[i]
    
    return s

def composition_from_formula(s):
    a=proper_split(s)
    b=join_numbers(a)
    
    brakets=[1]
    extracted=[]
    for symbol in b:
        if(symbol=='(' or symbol==')'or symbol=='[' or symbol==']'):
             extracted.append(symbol)
    for i in range(1,len(extracted)):
        if(extracted[i]=='[' and extracted[i-1]=='['):
            brakets.append(brakets[-1]+1)
        elif(extracted[i]=='[' and extracted[i-1]==']'):
            brakets.append(brakets[-1])
        elif(extracted[i]==']' and extracted[i-1]=='['):
            brakets.append(brakets[-1])
        elif(extracted[i]==']' and extracted[i-1]==']'):
            brakets.append(brakets[-1]-1)
        elif(extracted[i]=='(' and extracted[i-1]=='('):
            brakets.append(brakets[-1]+1)
        elif(extracted[i]=='(' and extracted[i-1]==')'):
            brakets.append(brakets[-1])
        elif(extracted[i]==')' and extracted[i-1]=='('):
            brakets.append(brakets[-1])
        elif(extracted[i]==')' and extracted[i-1]==')'):
            brakets.append(brakets[-1]-1)
        elif(extracted[i]=='(' and extracted[i-1]=='['):
            brakets.append(brakets[-1]+1)
        elif(extracted[i]=='[' and extracted[i-1]==')'):
            brakets.append(brakets[-1]+1)
        elif(extracted[i]==')' and extracted[i-1]==']'):
            brakets.append(brakets[-1]-1)
        elif(extracted[i]==']' and extracted[i-1]==')'):
            brakets.append(brakets[-1]-1)
        elif(extracted[i]==')' and extracted[i-1]=='['):
            brakets.append(brakets[-1])
        elif(extracted[i]==']' and extracted[i-1]=='('):
            brakets.append(brakets[-1])        
            
    i=0
    for j,elem in enumerate(b):
        if(elem=='[' or elem==']'):
            b[j]=b[j]+str(brakets[i])
            i+=1
        elif(elem=='(' or elem==')'):
            b[j]=b[j]+str(brakets[i])
            i+=1

    n=max(brakets)
    braket_layers={}
    for order in range(1,n+1):
        left=[]
        right=[]
        for i,symi in enumerate(b):
            if(symi=='('+str(order) or symi=='['+str(order)):
                left.append(i)
            elif(symi==')'+str(order) or symi==']'+str(order)):
                right.append(i)
        pairs=[]
        for i in range(len(left)):
            pairs.append((left[i],right[i]))
        braket_layers[order]=pairs
        
    list_of_elements=[]
    
    for i,elem in enumerate(b):
        if(re.search('[A-Za-z]',elem)):
            if(i<len(b)-1):
                if(re.search('[0-9]',b[i+1]) and not re.search('\(|\)|\[|\]',b[i+1])):
                    num=float(b[i+1])
                else:
                    num=1
            else:
                num=1
            list_of_elements.append((elem,i,num))
        
            
    for order in range(1,n+1):
        for coord in braket_layers[order]:
            if(coord[1]<len(b)-1):
                if(re.search('[0-9]',b[coord[1]+1]) and not re.search('\(|\)|\[|\]',b[coord[1]+1])):
                    factor=float(b[coord[1]+1])
                else:
                    factor=1
            else:
                factor=1
            for k,elem in enumerate(list_of_elements):
                if(elem[1]>coord[0] and elem[1]<coord[1]):
                    new_value=elem[2]*factor
                    name=elem[0]
                    position=elem[1]
                    list_of_elements[k]=(name,position,new_value)
                    
    names=[]
    for elem in list_of_elements:
        names.append(elem[0])
    names=set(names)
    
    composition={}
    for name in names:
        composition[name]=0
        for elem in list_of_elements:
            if(elem[0]==name):
                composition[name]+=round(elem[2],4)  
    return composition


def merged_comp(dis_data):
    compositions=[]
    for i in range(len(dis_data)):
        form=dis_data.iloc[i]['formula']
        try:
            comp=composition_from_formula(form)
            compositions.append(comp)
        except:
            print('composition error:',i,form)
    
    
    normalised_compositions=[]
    for comp in compositions:
        index=len(elem_list)-1
        switch=0
        for el in comp.keys():
            if el not in elem_list:
                switch=1   
        if(switch==0):
            for el in comp.keys():    
                if(elem_list.index(el)<index):
                    index=elem_list.index(el)
            devider=float(comp[elem_list[index]])
            for el in comp.keys(): 
                comp[el]=round(float(comp[el])/devider*10000)
            normalised_compositions.append(comp) 
        else:
            normalised_compositions.append({})
            
            
    merged_compositions=[]
    for comp in normalised_compositions:
        string=''
        order=[]
        for el in comp.keys():
            order.append(elem_list.index(el))
        order=np.sort(order)
        for index in order:
            string=string+elem_list[index]+str(comp[elem_list[index]])
        merged_compositions.append(string)
            
    return merged_compositions