import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

random.seed()
# random.uniform(mn,mx)
# random.gauss(mu,sigma) # mu = 0 and signma should be determined
velocityBound = pd.DataFrame([[0,-0.2,0,0],[20,0.2,10,0.01]],index=['min','max'],columns=['a0','alpha','ae','sigma'])
particleDimensionBound = pd.DataFrame([[0,-2,0,0],[400,2,70,0.1]],index=['min','max'],columns=['a0','alpha','ae','sigma'])
distanceIntensityDF = pd.read_csv('distance-soundIntensity-data.csv')
columns = ['a0','alpha','ae','sigma']

def computeIntensity(a0,alpha,d,ae,sigma):
    return (a0 * np.exp(-alpha * d) + ae) * (1 - np.random.normal(0,sigma,len(d)))

def computeFitness(x,distanceIntensityDF):
    a0,alpha,ae,sigma = x
    particleIntensityComputation = computeIntensity(a0,alpha,distanceIntensityDF['distance'],ae,sigma)
    errors = distanceIntensityDF['soundIntensity'] - particleIntensityComputation
    squareErrors = errors**2
    
    return sum(squareErrors)
def updatePersonalBest(pdSeries):
    if pdSeries['fitness'] < pdSeries['personalBestFitness']:
        pdSeries['personalBest'] = pdSeries['xi']
        pdSeries['personalBestFitness'] = pdSeries['fitness']
    return pdSeries
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
    
    df = pd.DataFrame(columns = ['xi','fitness','prevVelocity','personalBest','personalBestFitness','indexOfBestNeighbour',],index=np.arange(population))
    df['xi'] = [np.array([np.random.uniform(pDim.loc['min','a0'],pDim.loc['max','a0']),\
                  np.random.uniform(pDim.loc['min','alpha'],pDim.loc['max','alpha']),\
                  np.random.uniform(pDim.loc['min','ae'],pDim.loc['max','ae']),\
                  np.random.uniform(pDim.loc['min','sigma'],pDim.loc['max','sigma'])]) for i in np.arange(population)]#sigma
    df['prevVelocity'] = [np.array([0,0,0,0]) for i in np.arange(population)] # initial velocity of particles is 0 in all dimensions
    return df

def computeVelocity(pdSeries,bestNeighbour,velocityBound,w=0.7,c1=2,c2=0.5):
    minVelocity = velocityBound.loc['min',:].values
    maxVelocity = velocityBound.loc['max',:].values
    
    newVelocity = w * pdSeries['prevVelocity'] + c1 * (pdSeries['personalBest'] - pdSeries['xi']) + c2 * (bestNeighbour - pdSeries['xi'])
    
    #apply velocity boundaries
    for index,mn,v,mx in zip(range(len(newVelocity)), minVelocity, newVelocity, maxVelocity):
        if v < mn:
            newVelocity[index] = mn
        elif v > mx:
            newVelocity[index] = mx
    
    return newVelocity
def particleStep(pdSeries,particleDimensionBound):
    minBound = particleDimensionBound.loc['min',:].values
    maxBound = particleDimensionBound.loc['max',:].values
    velocity = pdSeries['prevVelocity']
    prevXi = pdSeries['xi']
    xi = prevXi + velocity
    
    for index,mn,i,mx in zip(range(len(xi)),minBound,xi,maxBound):
        if i < mn:
            xi[index] = mn
        elif i > mx:
            xi[index] = mx
    return xi

def psoSteps(population,particleDimensionBound,velocityBound,columns,distanceIntensityDF,w=0.7,c1=1.5,c2=1.2):
    #generate particles
    particlesDF = generateParticlesTuple(population,particleDimensionBound)
    particlesDF['fitness'] = particlesDF.apply(lambda x: computeFitness(x['xi'],distanceIntensityDF),axis=1)
    particlesDF['personalBest'] = particlesDF['xi']
    particlesDF['personalBestFitness'] = particlesDF['fitness']
    globalBest = particlesDF.loc[particlesDF['personalBestFitness'] == particlesDF['personalBestFitness'].min(),:]
    particlesDF['indexOfBestNeighbour'] = globalBest.index[0]
    gBestDF = pd.DataFrame(columns=['personalBest','fitness'])
    itCount = 0
    gBestDF.loc[itCount,:] = globalBest[['personalBest','fitness']].tail(1).values
    
   
    for i in range(100):
        #comput particle fitness
        particlesDF['fitness'] = particlesDF.apply(lambda x: computeFitness(x['xi'],distanceIntensityDF),axis=1)
        
        #update personalBests
        particlesDF = particlesDF.apply(lambda x: updatePersonalBest(x), axis=1)
        
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
        particlesDF['prevVelocity'] = particlesDF.apply(lambda x: computeVelocity(x,particlesDF.loc[x['indexOfBestNeighbour'],'personalBest'],velocityBound),axis=1)
        particlesDF['xi'] = particlesDF.apply(lambda x: particleStep(x,particleDimensionBound),axis=1)
        print('{}/{}: {}'.format(itCount,100,gBestDF['fitness'].iloc[-1]))
    gBestDF['fitness'].plot()
    f,ax = plt.subplots()
    distanceIntensityDF.plot(x='distance',y='soundIntensity',ax=ax,label='rawData')
    a0,alpha,ae,sigma = gBestDF['personalBest'].tail(1).iloc[0]
    
    gBestIntensities = computeIntensity(a0,alpha,distanceIntensityDF['distance'],ae,sigma)
    
    ax.plot(distanceIntensityDF['distance'],gBestIntensities,label='gBest')
    plt.title('({:.2f} * exp(-{:.2f} * d) + {:.2f}) * (1 - rand(0,{:.2f}))'.format(a0,alpha,ae,sigma))
    plt.legend()
    
    return particlesDF,gBestDF
#    maxNumOfIterations = 100
#    numOfIterations = 0
#    while numOfIterations <= maxNumOfIterations:
#        
    
f,gBest = psoSteps(100,particleDimensionBound,velocityBound,columns,distanceIntensityDF)