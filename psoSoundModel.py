import random
import numpy as np
import pandas as pd

random.seed()
# random.uniform(mn,mx)
# random.gauss(mu,sigma) # mu = 0 and signma should be determined
velocityBound = pd.DataFrame([[0,-0.2,0,0],[200,0.2,10,0.1]],index=['min','max'],columns=['a0','alpha','ae','sigma'])
particleDimensionBound = pd.DataFrame([[0,-2,0,0],[2000,2,100,1]],index=['min','max'],columns=['a0','alpha','ae','sigma'])
distanceIntensityDF = pd.read_csv('distance-soundIntensity-data.csv')
columns = ['a0','alpha','ae','sigma']

def computeIntensity(a0,alpha,d,ae,mu,sigma):
    return (a0 * np.exp(-alpha * d) + ae) * (1 - random.gauss(mu,sigma))

def computeFitness(a0,alpha,distanceIntensityDF,ae,mu,sigma):
    particleIntensityComputation = computeIntensity(a0,alpha,distanceIntensityDF['distance'],ae,mu,sigma)
    errors = distanceIntensityDF['soundIntensity'] - particleIntensityComputation
    squareErrors = errors**2
    
    return sum(squareErrors)

def generateParticles(population,particleDimensionBound,columns=['a0','alpha','ae','sigma']):
    pDim = particleDimensionBound
    
    df = pd.DataFrame(columns=columns,index=np.arange(population))
    df['a0'] = np.random.uniform(pDim.loc['min','a0'],pDim.loc['max','a0'],population)
    df['alpha'] = np.random.uniform(pDim.loc['min','alpha'],pDim.loc['max','alpha'],population)
    df['ae'] = np.random.uniform(pDim.loc['min','ae'],pDim.loc['max','ae'],population)
    df['sigma'] = np.random.uniform(pDim.loc['min','sigma'],pDim.loc['max','sigma'],population)
    return df
def generateParticlesTuple(population,particleDimensionBound):
    pDim = particleDimensionBound
    
    df = pd.DataFrame(columns = ['xi','fitness','personalBest','personalBestFitness','bestNeighbour'],index=np.arange(population))
    df['xi'] = [(np.random.uniform(pDim.loc['min','a0'],pDim.loc['max','a0']),\
                  np.random.uniform(pDim.loc['min','alpha'],pDim.loc['max','alpha']),\
                  np.random.uniform(pDim.loc['min','ae'],pDim.loc['max','ae']),\
                  np.random.uniform(pDim.loc['min','sigma'],pDim.loc['max','sigma'])) for i in np.arange(population)]#sigma
    return df
    
def psoSteps(population,particleDimensionBound,velocityBound,columns,distanceIntensityDF):
    #generate particles
#    particlesDF = generateParticles(population,particleDimensionBound,columns)
#    
#    particlesDF['fitness'] = particlesDF.apply(lambda x: computeFitness(x['a0'],x['alpha'],distanceIntensityDF,x['ae'],0,x['sigma']),axis=1)
    particlesDF = generateParticlesTuple(population,particleDimensionBound)
    particlesDF['fitness'] = particlesDF.apply(lambda x: computeFitness(x['xi'],distanceIntensityDF),axis=1)
    
    return particlesDF
#    maxNumOfIterations = 100
#    numOfIterations = 0
#    while numOfIterations <= maxNumOfIterations:
#        
    
f = psoSteps(100,particleDimensionBound,velocityBound,columns,distanceIntensityDF)