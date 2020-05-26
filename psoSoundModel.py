import random
import numpy as np
import pandas as pd

random.seed()
# random.uniform(mn,mx)
# random.gauss(mu,sigma) # mu = 0 and signma should be determined
velocityBound = pd.DataFrame([[0],[200]],index=['min','max'],columns=['K'])
particleDimensionBound = pd.DataFrame([[0],[2000]],index=['min','max'],columns=['K'])
distanceIntensityDF = pd.read_csv('distance-soundIntensity-data.csv')
columns = ['K']

def computeIntensity(K,d):
    return K / (d * d)
 
def computeFitness(K,distanceIntensityDF):
    particleIntensityComputation = computeIntensity(K,distanceIntensityDF['distance'])
    errors = distanceIntensityDF['soundIntensity'] - particleIntensityComputation
    squareErrors = errors**2
    
    return sum(squareErrors)

def generateParticles(population,particleDimensionBound,columns=['K']):
    pDim = particleDimensionBound
    
    df = pd.DataFrame(columns=columns,index=np.arange(population))
    df['K'] = np.random.uniform(pDim.loc['min','K'],pDim.loc['max','K'],population)
    return df
def generateParticlesTuple(population,particleDimensionBound):
    pDim = particleDimensionBound
    
    df = pd.DataFrame(columns = ['K','fitness','personalBest','personalBestFitness','indexOfBestNeighbour',],index=np.arange(population))
    df['K'] = [(np.random.uniform(pDim.loc['min','K'],pDim.loc['max','K'])])) for i in np.arange(population)]#sigma
    return df
    
def psoSteps(population,particleDimensionBound,velocityBound,columns,distanceIntensityDF):
    #generate particles
    particlesDF = generateParticlesTuple(population,particleDimensionBound)
    particlesDF['fitness'] = particlesDF.apply(lambda x: computeFitness(x['K'],distanceIntensityDF),aKs=1)
    particlesDF['personalBest'] = particlesDF['K']
    particlesDF['personalBestFitness'] = particlesDF['fitness']
    globalBest = particlesDF.loc[particlesDF['personalBestFitness'] == particlesDF['personalBestFitness'].min(),:]
    particlesDF['indexOfBestNeighbour'] = globalBest.index[0]
    return particlesDF
#    maxNumOfIterations = 100
#    numOfIterations = 0
#    while numOfIterations <= maxNumOfIterations:
#        
    
f = psoSteps(100,particleDimensionBound,velocityBound,columns,distanceIntensityDF)