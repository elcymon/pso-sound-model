import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

random.seed()
# random.uniform(mn,mx)
# random.gauss(mu,sigma) # mu = 0 and signma should be determined
velocityBound = pd.DataFrame([[0],[200]],index=['min','max'],columns=['K'])
particleDimensionBound = pd.DataFrame([[0],[2000]],index=['min','max'],columns=['K'])
distanceIntensityDF = pd.read_csv('distance-soundIntensity-data.csv')
columns = ['K']

def computeIntensity(K,d):
    return K / (d+0.3)
 
def computeFitness(K,distanceIntensityDF):
    particleIntensityComputation = computeIntensity(K,distanceIntensityDF['distance'])
    errors = distanceIntensityDF['soundIntensity'] - particleIntensityComputation
    squareErrors = errors**2
    
    return sum(squareErrors)
def updatePersonalBest(pdSeries):
    if pdSeries['fitness'] < pdSeries['personalBestFitness']:
        pdSeries['personalBest'] = pdSeries['K']
        pdSeries['personalBestFitness'] = pdSeries['fitness']
    return pdSeries

def generateParticles(population,particleDimensionBound,columns=['K']):
    pDim = particleDimensionBound
    
    df = pd.DataFrame(columns=columns,index=np.arange(population))
    df['K'] = np.random.uniform(pDim.loc['min','K'],pDim.loc['max','K'],population)
    return df
def generateParticlesTuple(population,particleDimensionBound):
    pDim = particleDimensionBound
    
    df = pd.DataFrame(columns = ['K','fitness','personalBest','personalBestFitness','indexOfBestNeighbour',],index=np.arange(population))
    df['K'] = [(np.random.uniform(pDim.loc['min','K'],pDim.loc['max','K'])) for i in np.arange(population)]#sigma
    return df

def computeVelocity(pdSeries,bestNeighbour,velocityBound,w=0.7,c1=2,c2=0.5):
    minVelocity = velocityBound.loc['min',:].values
    maxVelocity = velocityBound.loc['max',:].values
    # print(minVelocity,maxVelocity,pdSeries['prevVelocity'],pdSeries['personalBest'] , pdSeries['K'], bestNeighbour)
    newVelocity = w * pdSeries['prevVelocity'] + c1 * (pdSeries['personalBest'] - pdSeries['K']) + c2 * (bestNeighbour - pdSeries['K'])
    
    #apply velocity boundaries

    
    if newVelocity < minVelocity:
        newVelocity = minVelocity
    elif newVelocity > maxVelocity:
        newVelocity = maxVelocity
    
    return newVelocity
def particleStep(pdSeries,particleDimensionBound):
    minBound = particleDimensionBound.loc['min',:].values
    maxBound = particleDimensionBound.loc['max',:].values
    velocity = pdSeries['prevVelocity']
    prevK = pdSeries['K']
    K = prevK + velocity
    
    if K < minBound:
        K = minBound
    elif K > maxBound:
        K = maxBound
    return K

def psoSteps(population,particleDimensionBound,velocityBound,columns,distanceIntensityDF,w=0.7,c1=1.5,c2=1.2):
    #generate particles
    particlesDF = generateParticlesTuple(population,particleDimensionBound)
    
    particlesDF['fitness'] = particlesDF.apply(lambda x: computeFitness(x['K'],distanceIntensityDF),axis=1)
    particlesDF['personalBest'] = particlesDF['K']
    particlesDF['personalBestFitness'] = particlesDF['fitness']
    globalBest = particlesDF.loc[particlesDF['personalBestFitness'] == particlesDF['personalBestFitness'].min(),:]
    particlesDF['indexOfBestNeighbour'] = globalBest.index[0]
    gBestDF = pd.DataFrame(columns=['personalBest','fitness'])
    itCount = 0
    gBestDF.loc[itCount,:] = globalBest[['personalBest','fitness']].tail(1).values
    particlesDF['prevVelocity'] = 0
   
    for i in range(100):
        #comput particle fitness
        particlesDF['fitness'] = particlesDF.apply(lambda x: computeFitness(x['K'],distanceIntensityDF),axis=1)
        
        #update personalBests
        particlesDF = particlesDF.apply(lambda x: updatePersonalBest(x),axis=1)
        
        #update global best
        globalBest = particlesDF.loc[particlesDF['personalBestFitness'] == particlesDF['personalBestFitness'].min(),:]
#        print(gBestDF['fitness'].tail(1))
        if globalBest['fitness'].iloc[0] < gBestDF['fitness'].tail(1).iloc[0]:
            
            gBestDF.loc[itCount,:] = globalBest[['personalBest','fitness']].tail(1).values
            particlesDF['indexOfBestNeighbour'] = globalBest.index[0]
        else:
            gBestDF.loc[itCount,:] = gBestDF[['personalBest','fitness']].tail(1).values
        itCount += 1    
        
        
        #apply velocities
        # print(particlesDF)
        particlesDF['prevVelocity'] = particlesDF.apply(lambda x: computeVelocity(x,particlesDF.loc[x['indexOfBestNeighbour'],'personalBest'],velocityBound),axis=1)
        particlesDF['K'] = particlesDF.apply(lambda x: particleStep(x,particleDimensionBound),axis=1)
        print('{}/{}: {}'.format(itCount,100,gBestDF['fitness'].iloc[-1]))
    gBestDF['fitness'].plot()
    f,ax = plt.subplots()
    distanceIntensityDF.plot(x='distance',y='soundIntensity',ax=ax,label='rawData')
    K = gBestDF['personalBest'].tail(1).iloc[0]
    
    gBestIntensities = computeIntensity(K,distanceIntensityDF['distance'])
    
    ax.plot(distanceIntensityDF['distance'],gBestIntensities,label='gBest')
    plt.title('({:.4f}/d^2))'.format(K))
    plt.legend()
    plt.show()
    return particlesDF,gBestDF
#    maxNumOfIterations = 100
#    numOfIterations = 0
#    while numOfIterations <= maxNumOfIterations:
#        
    
f,gBest = psoSteps(100,particleDimensionBound,velocityBound,columns,distanceIntensityDF)