import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import scale


def population(size,lower_bound,upper_bound,mean,std):
    
    #size: number of individuals in the population

    #lb: lower bound of talent value
    lb = np.full(size,lower_bound)

    #ub: upper bound of talent value
    ub = np.full(size,upper_bound)

    #mu: mean of talent distribution
    mu = np.full(size,mean)
   
    #sigma: standard deviation of talent distribution
    sigma = np.full(size,std)

    talent = np.zeros(size)
    talent = sp.stats.truncnorm.rvs((lb-mu)/sigma, (ub-mu)/sigma, loc=mean, scale=std) 

    mean = np.mean(talent)
    std = np.std(talent)
    
    talent_sort = np.sort(talent,0,'quicksort')
    talent_index = np.argsort(talent,0,'quicksort')

    return talent_sort, talent_index


def evolution(talent,time,unlucky_event,lucky_event):

    rng = default_rng()

    capital = np.full((len(talent),time),10.0)

    for i in range(time-1):
        for j in range(len(talent)):

            a = rng.uniform(0.0,1.0)

            if a <=unlucky_event:

                capital[j,i+1] = capital[j,i]/2
            
            elif a<= unlucky_event + lucky_event:
                
                b = rng.uniform(0.0,1.0)

                if b<talent[j]:
                    capital[j,i+1] = capital[j,i]*2

                else:
                    capital[j,i+1] = capital[j,i]

            else:
                capital[j,i+1] = capital[j,i]

    return capital