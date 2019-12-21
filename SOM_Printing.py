# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 09:50:48 2019

@author: oakl
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

# def diff_graph(show=False, printout=True, returns=False, path='./'):

#     """Plot a 2D map with nodes and weights difference among neighbouring nodes.

#     Args:
#         show (bool, optional): Choose to display the plot.
#         printout (bool, optional): Choose to save the plot to a file.
#         returns (bool, optional): Choose to return the difference value.

#     Returns:
#         (list): difference value for each node.             
#     """
    
#     neighbours=[]
#     for node in self.nodeList:
#         nodelist=[]
#         for nodet in self.nodeList:
#             if node != nodet and node.get_nodeDistance(nodet) <= 1.001:
#                 nodelist.append(nodet)
#         neighbours.append(nodelist)     
        
#     diffs = []
#     for node, neighbours in zip(self.nodeList, neighbours):
#         diff=0
#         for nb in neighbours:
#             diff=diff+node.get_distance(nb.weights)
#         diffs.append(diff)  

#     centers = [[node.pos[0],node.pos[1]] for node in self.nodeList]

#     if show==True or printout==True:
    
#         widthP=100
#         dpi=72
#         xInch = self.netWidth*widthP/dpi 
#         yInch = self.netHeight*widthP/dpi 
#         fig=plt.figure(figsize=(xInch, yInch), dpi=dpi)

#         ax = hx.plot_hex(fig, centers, diffs)
#         ax.set_title('Nodes Grid w Weights Difference', size=80)
        
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.0)
#         cbar=plt.colorbar(ax.collections[0], cax=cax)
#         cbar.set_label('Weights Difference', size=80, labelpad=50)
#         cbar.ax.tick_params(labelsize=60)
#         plt.sca(ax)

#         printName=os.path.join(path,'nodesDifference.png')
        
#         if printout==True:
#             plt.savefig(printName, bbox_inches='tight', dpi=dpi)
#         if show==True:
#             plt.show()
#         if show!=False and printout!=False:
#             plt.clf()

#     if returns==True:
#         return diffs 


# This code for distance_plotter comes from Ross's machine_learning notebook
def distance_plotter(attributes_df, som, labels=None):
    # setup figure
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # correct weightings of edge points (for lack of neighbours)
    mid = som.distance_map()
    mid[:,0] = np.minimum(1, mid[:,0]*1.5)
    mid[:,len(mid[0,:]) - 1] = np.minimum(1, mid[:,len(mid[0,:]) - 1]*1.5)
    mid[0,:] = np.minimum(1, mid[0,:]*1.5)
    mid[len(mid[:,0]) - 1,:] = np.minimum(1, mid[len(mid[:,0]) - 1,:]*1.5)

    # create heatmap of mean inter-neuron distance
    plt.pcolor(mid.T, norm=colors.PowerNorm(gamma=1./2), cmap='RdPu', vmin=np.quantile(mid, 0.01), vmax=1)
    cb = plt.colorbar()
    cb.set_label('Mean inter-neuron distance')
    
    # calculate number of events in each neuron
    if labels is None:
        count = np.zeros_like(mid)
        X = attributes_df.iloc[:, 2:len(attributes_df.columns)]
        for x in X.values:
            w = som.winner(x)
            count[w[0], w[1]] += 1
        
        # plot circles over heatmap to represent number of events per neuron
        plt.plot(1,1)
        for i in range(0, len(count[:,0])):
            for j in range(0, len(count[0,:])):
                plt.plot(i + 0.5, j + 0.5, marker = 'o', markeredgecolor='black', markerfacecolor = 'None', \
                                markersize = 400*np.sqrt(count[i, j]/np.max(count))/len(count[0,:]))
    else:
        # plot contour map of mean inter-neuron distance
        for i in range(0, int(np.max(labels)) + 1):
            plt.contour(np.minimum(np.absolute(labels.T - i), 0.5), colors='k', levels=0.15, \
                            linewidths=1, origin='lower')
            
        # add labels to each cluster
        locations = get_locations(som)
        for i in range(0, int(np.max(labels)) + 1):
            # find median x and y location for cluster
            result = np.where(labels == i)
            xs = [x[0] for x in locations[result]]
            ys = [x[1] for x in locations[result]]
            plt.annotate('cluster '+str(i + 1), xy=(np.median(xs) + 0.5, np.median(ys) + 0.5), \
                            horizontalalignment='center', verticalalignment='middle', size=9.5, \
                            bbox=dict(boxstyle="round", alpha=0.5, fc=(1.0, 1.0, 1.0), ec="none"))