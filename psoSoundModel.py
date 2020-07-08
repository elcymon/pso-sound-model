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
    f,ax = plt.subplots(figsize=(6,3))
    cmap = mpl.cm.get_cmap('inferno')
    sample = np.linspace(distanceIntensityDF.index[0],distanceIntensityDF.index[-1],1000,dtype=np.int)
    distanceIntensityDF.loc[sample,:].plot(x='distance',y='soundIntensity',linestyle='',marker='o',
                           markerfacecolor='none',markeredgecolor=cmap(0.1),
                           ax=ax,label='Raw Data',alpha=1)
    a0 = 299.1795
    alpha = 0.1039
    ae = 48.1824
    sigma = 0.06
    modeleqn = r'$A = {:.4f} \times e^{{-{:.4f} \times d}} + {:.4f}$'.format(a0,alpha,ae)#'A = ({:.2f} * exp(-{:.2f} * d) + {:.2f}) * (1 - rand(0,{:.2f}))'.format(a0,alpha,ae,sigma)
    modeleqn  = r'$A = A_{0} e^{-\alpha d} + A_e$'
    modeleqn = r'$A = \left(A_{0} e^{-\alpha d} + A_e\right) \times \left(1 - N(\mu,\sigma^2)\right)$'
    gBestIntensities = computeIntensity(a0,alpha,distanceIntensityDF['distance'],ae,sigma)
    ax.plot(distanceIntensityDF.loc[sample,'distance'],gBestIntensities[sample],label=modeleqn,linestyle='',marker='s',markerfacecolor='none',markeredgecolor=cmap(0.6),alpha=0.6)
    ax.set_xlabel('Distance in metres',fontsize=12,fontweight='bold')
    ax.set_ylabel('Sound Intensity',fontsize=12,fontweight='bold')
    ax.set_ylim([0,500])
    # plt.title('({:.2f} * exp(-{:.2f} * d) + {:.2f}) * (1 - rand(0,{:.2f}))'.format(a0,alpha,ae,sigma))
    plt.legend(fontsize=12,frameon=True,edgecolor='k')
    f.savefig('noisy_model.pdf',bbox_inches='tight',dpi=100)
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
def txt2csv(resultsfolder,fnames,
            columns=['t','distance','drop1','drop2','drop3','soundIntensity'],
            keep=['distance','soundIntensity']):
    for f in fnames:
        df = pd.read_csv(f'{resultsfolder}/{f}.txt',names=columns)
        df[keep].to_csv(f'{resultsfolder}/{f}.csv')

def plotGradient():
    resultsfolder = 'sound-experiment-data'
    fname = '20180909231249datagoexp1a'
    qsizes = [1,8,20,40,80,120]
    df = pd.read_csv(f'{resultsfolder}/{fname}.csv',index_col=0)
    
#    return df
#    df.sort_index(axis=0,ascending=False,inplace=True)
    
    
    for q in qsizes:
        print(q)
        gradient = pd.DataFrame(index=df.index,columns=['distance','soundIntensity','gradient'])
        gradient['distance'] = df['distance']
        f,ax = plt.subplots(figsize=(4,3))
#        if q == 1:
#            gradient['gradient'] = df['soundIntensity'].diff()
#            gradient['distance'] = df['distance']
#            gradient['soundIntensity'] = df['soundIntensity']
#            return gradient
        i = 0
        istop = i + q - 1
        prev_sound = np.nan
        prev_stop = np.nan
        plusgradient = None
        neggradient = None
        while istop < df.index[-1]:
            if i + q - 1 < df.index[-1]:
                istop = i + q -1
            else:
                istop = df.index[-1]
                
            curr_sound = df.loc[i:istop,'soundIntensity'].mean()
            gradient.loc[i:istop,'soundIntensity'] = curr_sound
            gradient.loc[i:istop,'gradient'] =  prev_sound - curr_sound
            
            if not np.isnan(prev_stop):
                x = gradient.loc[[prev_stop, istop],'distance']
                y = [prev_sound, curr_sound]
                if prev_sound - curr_sound > 0:
                    ax.plot(x,y,'b-')
                    if plusgradient is None:
                        plusgradient = [x,y]
                elif prev_sound - curr_sound < 0:
                    ax.plot(x,y,'r-')
                    if neggradient is None:
                        neggradient = [x,y]
            
            prev_sound = curr_sound
            prev_stop = istop
            i = i + q
            
            
        ax.set_xlabel('Distance in metres',fontweight='bold')
        ax.set_ylabel('Sound Intensity',fontweight='bold')
        ax.set_ylim([0,500])
        totalplus = (gradient['gradient'] > 0).sum()
        totalneg = (gradient['gradient'] < 0).sum()
        ax.plot(*plusgradient,'b-',label=f'{totalplus/(totalplus+totalneg)*100:.0f}%')
        ax.plot(*neggradient,'r-',label=f'{totalneg/(totalplus+totalneg)*100:.0f}%')
        plt.legend(fontsize=14)
        f.savefig(f'{resultsfolder}/{fname}-q{q}.pdf',bbox_inches='tight')
def computeGradient(fname,q):
    if fname is None:
        a0 = 299.1795
        alpha = 0.1039
        ae = 48.1824
        sigma = 0.06
        df = pd.DataFrame(index=np.arange(8000),columns=['distance','soundIntensity'])
        df['distance'] = np.linspace(0,15,8000)
        df['soundIntensity'] = computeIntensity(a0,alpha,df['distance'],ae,sigma)
#        df.plot(x='distance',y='soundIntensity')
#        return
    else:
        df = pd.read_csv(fname,index_col=0)
    
    gradient = pd.DataFrame(index=df.index,columns=['distance','soundIntensity','gradient'])
    gradient['distance'] = df['distance']
    if q == 1:
        gradient['soundIntensity'] = df['soundIntensity']
        gradient['gradient'] = df['soundIntensity'].sort_index(ascending=False)\
                                    .diff().sort_index(ascending=True)
    else:
        i = 0
        istop = i + q - 1
        prev_sound = np.nan
        
        while istop < df.index[-1]:
            if i + q - 1 < df.index[-1]:
                istop = i + q -1
            else:
                istop = df.index[-1]
                
            curr_sound = df.loc[i:istop,'soundIntensity'].mean()
            gradient.loc[i:istop,'soundIntensity'] = curr_sound
            gradient.loc[i:istop,'gradient'] =  prev_sound - curr_sound
            
            
            prev_sound = curr_sound
            
            i = i + q
        
    totalplus = (gradient['gradient'] > 0).sum()
    totalneg = (gradient['gradient'] < 0).sum()
    return totalplus/(totalplus+totalneg)*100, totalneg/(totalplus+totalneg)*100

def barplotGradient():
    resultsfolder = 'sound-experiment-data'
    qsizes = [1,8,20,40,80,120]
    columns = pd.MultiIndex.from_product([['Raw Data','Model Data'],['Positive','Negative'],['Mean','Stddev']])
    bardata = pd.DataFrame(index=qsizes,columns=columns)
    cmap = mpl.cm.get_cmap('inferno')
    f,ax = plt.subplots(figsize=(6,3))
    files = ['20180909231249datagoexp1a','20180909231651datagoexp1b',
            '20180909232055data','20180909232518data','20180909232933data']
    for  q in qsizes:
        print(q)
        pluslist = []
        neglist = []
        for fname in files:
            pctplus,pctneg = computeGradient(f'{resultsfolder}/{fname}.csv',q)
            pluslist.append(pctplus)
            neglist.append(pctneg)
        
        bardata.loc[q,[('Raw Data','Positive','Mean')]] = round(np.mean(pluslist),1)
        bardata.loc[q,[('Raw Data','Negative','Mean')]] = round(np.mean(neglist),1)
        bardata.loc[q,[('Raw Data','Positive','Stddev')]] = round(np.std(pluslist),1)
        bardata.loc[q,[('Raw Data','Negative','Stddev')]] = round(np.std(neglist),1)
        
        pluslist = []
        neglist = []
        for i in range(5):
            pctplus,pctneg = computeGradient(None,q)
            pluslist.append(pctplus)
            neglist.append(pctneg)
        bardata.loc[q,[('Model Data','Positive','Mean')]] = round(np.mean(pluslist),1)
        bardata.loc[q,[('Model Data','Negative','Mean')]] = round(np.mean(neglist),1)
        bardata.loc[q,[('Model Data','Positive','Stddev')]] = round(np.std(pluslist),1)
        bardata.loc[q,[('Model Data','Negative','Stddev')]] = round(np.std(neglist),1)
#    return bardata    
    yerr = [[bardata[[('Raw Data','Positive','Stddev')]].values.ravel(),
                 bardata[[('Raw Data','Positive','Stddev')]].values.ravel()],
                [bardata[[('Raw Data','Negative','Stddev')]].values.ravel(),
                     bardata[[('Raw Data','Negative','Stddev')]].values.ravel()]]
    bardata[[('Raw Data','Positive','Mean'),('Raw Data','Negative','Mean')]]\
    .plot.bar(stacked=True, width=0.35, position=1,  color=[cmap(0.2),cmap(0.6)],
                capsize=4,ax=ax, alpha=1,yerr=yerr)
    
    yerr = [[bardata[[('Model Data','Positive','Stddev')]].values.ravel(),
                 bardata[[('Model Data','Positive','Stddev')]].values.ravel()],
                [bardata[[('Model Data','Negative','Stddev')]].values.ravel(),
                     bardata[[('Model Data','Negative','Stddev')]].values.ravel()]]
    bardata[[('Model Data','Positive','Mean'),('Model Data','Negative','Mean')]]\
    .plot.bar(stacked=True, width=0.35, position=0, color=[cmap(0.2),cmap(0.6)], ax=ax,
              capsize=4,alpha=0.6,yerr=yerr,rot=0)
    plt.legend(['Raw Data ($+\Delta>0$)','Raw Data ($+\Delta<0$)',
                'Model Data ($-\Delta>0$)','Model Data ($-\Delta<0$)'],
            fontsize=12,loc='upper right',bbox_to_anchor=(1.53,1))
    ax.set_xlim(-0.5,5.6)
    ax.set_xlabel('Queue size',fontsize=12,fontweight='bold')
    ax.set_ylabel('Percentage',fontsize=12,fontweight='bold')
    
    f.savefig(f'{resultsfolder}/model_vs_raw_data.pdf',bbox_inches='tight')
    
    return bardata
        
    
if __name__ == '__main__':
    txt2csv('sound-experiment-data',['20180909231249datagoexp1a','20180909231651datagoexp1b',
                                     '20180909232055data','20180909232518data','20180909232933data'])
    filename = 'sound-experiment-data/distance-soundIntensity-data.csv'
#    df = barplotGradient()
    distanceIntensityDF = pd.read_csv(filename)
    plotData(distanceIntensityDF)
#    dfDict = {'Directional' : 'sound-experiment-data/distance-soundIntensity-data.csv',
#              'Omnidirectional' : 'sound-experiment-data/20180901162100dataomnispeaker.csv',
#              #'Ambient Noise' : 'sound-experiment-data/20180901161305dataAmbientNoise.csv'
#              }
#    plotDataDictionary(dfDict)