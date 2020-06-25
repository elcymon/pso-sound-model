import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
random.seed()
# random.uniform(mn,mx)
# random.gauss(mu,sigma) # mu = 0 and signma should be determined

def computeIntensity(a0,alpha,d,ae,sigma):
    return (a0 * np.exp(-alpha * d) + ae) * (1 - np.random.normal(0,sigma,len(d)))


def plotData(distanceIntensityDF):
    f,ax = plt.subplots()
    cmap = mpl.cm.get_cmap('inferno')
    sample = np.linspace(distanceIntensityDF.index[0],distanceIntensityDF.index[-1],1000,dtype=np.int)
    distanceIntensityDF.loc[sample,:].plot(x='distance',y='soundIntensity',linestyle='',marker='o',markerfacecolor='w',markeredgecolor=cmap(0.1),ax=ax,label='Raw Data',alpha=0.6)
    a0 = 299.1795
    alpha = 0.1039
    ae = 48.1824
    sigma = 0#.06
    modeleqn = r'$A = {:.4f} \times e^{{-{:.4f} \times d}} + {:.4f}$'.format(a0,alpha,ae)#'A = ({:.2f} * exp(-{:.2f} * d) + {:.2f}) * (1 - rand(0,{:.2f}))'.format(a0,alpha,ae,sigma)
    modeleqn  = r'$A = A_{0} e^{-\alpha d} + A_e$'
#    modeleqn = r'$A = \left(A_{0} e^{-\alpha d} + A_e\right) \times \left(1 - N(\mu,\sigma^2)\right)$'
    gBestIntensities = computeIntensity(a0,alpha,distanceIntensityDF['distance'],ae,sigma)
    ax.plot(distanceIntensityDF.loc[sample,'distance'],gBestIntensities[sample],label=modeleqn,linestyle='',marker='s',markerfacecolor='none',markeredgecolor=cmap(0.6),alpha=0.6)
    ax.set_xlabel('Distance in metres')
    ax.set_ylabel('Sound Intensity')
    ax.set_ylim([0,500])
    # plt.title('({:.2f} * exp(-{:.2f} * d) + {:.2f}) * (1 - rand(0,{:.2f}))'.format(a0,alpha,ae,sigma))
    plt.legend()
    f.savefig('noisy_model.png',bbox_inches='tight',dpi=100)
    plt.show()

def plotDataDictionary(dfDict):
    f,ax = plt.subplots()
    cmap = mpl.cm.get_cmap('inferno')
    figname = ''
    for color,(label,filename) in enumerate(dfDict.items()):
        df = pd.read_csv(filename)
        sample = np.linspace(df.index[0],df.index[-1],1000,dtype=np.int)
        df.loc[sample,:].plot(x='distance',y='soundIntensity',linestyle='',marker='o',
              markerfacecolor=(0,0,0,0),markeredgecolor=cmap(color * 50),ax=ax,label=label)
        if len(figname) == 0:
            figname += label.replace(' ','')
        else:
            figname += '-' + label.replace(' ','')
    
    ax.set_xlabel('Distance in metres',fontweight='bold')
    ax.set_ylabel('Sound Intensity',fontweight='bold')
    ax.set_ylim([0,500])
    plt.legend()
    f.savefig(figname + '.png',bbox_inches='tight',dpi=100)
    plt.show()
if __name__ == '__main__':
    filename = 'sound-experiment-data/distance-soundIntensity-data.csv'
#    distanceIntensityDF = pd.read_csv(filename)
#    plotData(distanceIntensityDF)
    dfDict = {'Directional' : 'sound-experiment-data/distance-soundIntensity-data.csv',
              'Omnidirectional' : 'sound-experiment-data/20180901162100dataomnispeaker.csv',
              #'Ambient Noise' : 'sound-experiment-data/20180901161305dataAmbientNoise.csv'
              }
    plotDataDictionary(dfDict)