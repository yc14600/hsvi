import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_next_batch(data, B, ii,labels=None,repeat=True):
    if B == data.shape[0] or (not repeat and B > data.shape[0]):
        return data,labels,ii
    elif B > data.shape[0]:        
        n = B/data.shape[0]
        r = B%data.shape[0]
        x_batch = np.concatenate([np.repeat(data,n,axis=0),data[:r]])

        if labels is not None:
            y_batch = np.concatenate([np.repeat(labels,n,axis=0),labels[:r]])
            return x_batch,y_batch,0
        else:
            return x_batch,None,0

    if ii+B < data.shape[0]:
        if not labels is None:
            return data[ii:ii+B],labels[ii:ii+B],ii+B
            
        else:
            return data[ii:ii+B],None,ii+B
    else:
        if repeat:
            
            r = ii+B-data.shape[0]
            ids = np.arange(data.shape[0])
            batch = data[(ids>=ii)|(ids<r)]
            if labels is None:
                return batch,None,r
            else:
                return batch,labels[(ids>=ii)|(ids<r)],r
        else:
            if labels is None:
                return data[ii:],None,0
            else:
                return data[ii:],labels[ii:],0


def plot(samples,shape=None,cmap='Greys_r'):
    if shape is None:
        rows = 4
        cols = 4
    else:
        rows = shape[0]
        cols = shape[1]
        
    fig = plt.figure(figsize=(rows, cols))
    gs = gridspec.GridSpec(rows, cols)    

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])

        if cmap == 'Greys_r':
            assert(len(sample.shape)<3 or sample.shape[2]==1)
            sample = sample.reshape(sample.shape[0],sample.shape[1])

        plt.imshow(sample,cmap=cmap)

    return fig
