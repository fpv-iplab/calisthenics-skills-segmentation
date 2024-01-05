"""This Python module implements the code related to the following scientific paper:
    
A. Furnari, G. M. Farinella, S. Battiato, Personal-Location-Based Temporal Segmentation of Egocentric Video for Lifelogging Applications, submitted to Journal of Visual Communication and Image Representation

More information is available at our web page: http://iplab.dmi.unict.it/PersonalLocationSegmentation

The module is designed to work with Python 2.7"""
from __future__ import division
import numpy as np
import sys
from scipy.stats import mode as mode
from scipy.optimize import linear_sum_assignment

'''
def rejectNegatives(probs,K=300):
    """Performs rejection of negative samples using the sliding 
        window approach described in the paper.
        
        Args:
            probs: N X (M-1) matrix, where N is the number of samples
                and M-1 is the number of positive classes
            K: neighborhood size to perform rejection
        Returns:
            a N X M matrix of probabilities over the M classes (positive + negative classes)."""
    predictions = probs.argmax(1)
    augmented_probs = np.zeros((probs.shape[0],probs.shape[1]+1))
    augmented_probs[0:K,0:probs.shape[1]] = probs[0:K,:]
    for i in range(K,len(predictions)):
        window = predictions[i-K:i]

        c=mode(window).count[0]
        pos_prob=float(c)/len(window)
        pr=probs[i]*pos_prob

        augmented_probs[i,0:probs.shape[1]]=pr
        augmented_probs[i,probs.shape[1]]=1-pos_prob
    return augmented_probs

'''



def viterbi(emission_matrix, threshold, transition_matrix=None):
    """Implements the Viterbi algorithm.
    Args:
        emission_matrix: N x M matrix of probabilities, where N
            is the number of samples and M is the number of classes
        transition_matrix: M x M matrix of state transition. If None
            the almost identical matrix discussed in the paper will be used
    Returns:
        N-dimensional array of most likely states (i.e., labels predictions)"""
    #K: number of states
    K = emission_matrix.shape[1]
    #T: number of observations
    T = len(emission_matrix)
    
    if transition_matrix is None:
        transition_matrix = np.eye(K) + threshold
        transition_matrix/=transition_matrix.sum(1).reshape((-1,1))
    emission_matrix=np.log(emission_matrix+0.0000001)
    transition_matrix=np.log(transition_matrix)
    
    #consider a uniform distribution 
    pi = np.ones((K))
    pi/=pi.sum()
    
    #states from zero to T-1
    states = range(K)
    #initialize tables
    T1 = np.zeros((K,T))
    T2 = np.zeros((K,T),dtype=int)
    
    #for each state
    for s in states:
        #initialize values at zero
        T1[s,0] = pi[s]*emission_matrix[0,s]
        T2[s,0] = 0

    #for each observation
    for i in range(1,T):
        #for each state
        for s in states:
            w=T1[:,i-1]+transition_matrix[:,s]
            T1[s,i] = np.max(w)
            T1[s,i] += emission_matrix[i,s]
            T2[s,i] = np.argmax(w)
#            print w
    
    X=np.zeros((T))
    Z=np.zeros((T))
    
    Z[T-1]=np.argmax(T1[:,T-1])
    X[T-1]=Z[T-1]
    
    for i in range(1,T)[::-1]:
        Z[i-1]=T2[int(Z[i]),i]
        X[i-1]=Z[i-1]

    return X.astype(int)
    
    
def lab2seg(labels):
    """Converts arrays of labels into a list of video segments (considering connected components)
    Returns a Nx3 array. Each line s is a different segment such that:
        s[0] is the starting frame
        s[1] is the ending frame
        s[2] is the class label"""
    labels=np.array(labels)
    n_frames=len(labels)
    
    discontinuities=np.where(np.diff(labels))[0] #finds points where lables change in the array
                             #these are last indices of a homogenous segment
        
    start_idxs = np.concatenate(([0],discontinuities+1))  #finds starting indices (i.e., index 0
                                                        #plus index right after discontinuity)
    end_idxs = np.concatenate((discontinuities,[n_frames-1])) #finds ending indices (all discontinuities
                                                         #plus end frame)
    
    classes=labels[end_idxs] #gets classes corresponding to segments

    segments=np.vstack((start_idxs,end_idxs,classes)) 

    return segments.T
    
def jaccardIndex(s1,s2):
    """Computes the Jaccard index (i.e., Intersection Over Union (IOU) area) between two segments"""
    start_intersection = np.max((s1[0],s2[0]))
    end_intersection = np.min((s1[1],s2[1]))
    
    start_union = np.min((s1[0],s2[0]))
    end_union = np.max((s1[1],s2[1]))
    
    return (np.max(end_intersection-start_intersection+1,0))/(end_union-start_union+1)
        
def SF1(gt_labels,predicted_labels,curves='average',average='perclass',thresholds=np.linspace(0,1,100),labels=None):
    """Computes the segment-based Threshold-F1 curves.
    #x = linspace(0,1,100)
    #y = rsiultato
    Args:
        gt_labels: the array of ground truth labels
        predicted_labels: the array of predicted labels
        curves: a flag specifying how curves should be processed. 
            If curves='raw' a M x T matrix (M is the number of classes, T the number of thresholds) 
                of curves will be returned (one curve for each class)
            If curves='average' the average curve will be returned (average over classes)
            If curves=None no curves will be returned
        average: a flag specifying wheter the average SF1 scores should be computed
            If average='perclass', the list of all SF1 scores per class will be returned 
            If average='mean', the mean average SF1 score (over all classes) will be return 
            If average=None, no average SF1 value will be returned
        thresholds: list of thresholds to be used to compute the curves
        labels: optional list of class labels to be considered
        
    Returns:
        curves: an array (or a matrix) of threshold-SF1 scores depending on the parameters 
            and the average SF1 values will be returned as a second output
            some output values might be ommited depending on parameters
        """
        
    predicted_segmentation = lab2seg(predicted_labels)
    gt_segmentation = lab2seg(gt_labels)
    gt_classes = np.array([s[2] for s in gt_segmentation])
    predicted_classes = np.array([s[2] for s in predicted_segmentation])
    
    if labels is None:
        classes = np.unique(gt_classes)
    else:
        classes=labels
    
    curv=np.zeros((len(classes),len(thresholds)))
    
    for h,c in enumerate(classes):
        prs = predicted_segmentation[predicted_classes==c]
        gts = gt_segmentation[gt_classes==c]

        costs = np.zeros((len(gts),len(prs)))
        
        for i,gs in enumerate(gts):
            for j,ps in enumerate(prs):
                costs[i,j] = 1-jaccardIndex(gs,ps)
        
        row_ind,col_ind = linear_sum_assignment(costs)
        
        jaccard_indices=1-costs
        jaccard_indices=jaccard_indices[row_ind,col_ind]

        n_retrieved = len(prs)
        n_relevant = len(gts)        

        for k,t in enumerate(thresholds):
            tp = (jaccard_indices>=t).sum()
              
            precision = tp/n_retrieved if n_retrieved!=0 else 0
            recall = tp/n_relevant if n_relevant!=0 else 0
            
            curv[h,k] = 2*(precision*recall)/(precision+recall) if precision+recall!=0 else 0



    
    results=list()

    if curves=='average':
        results.append(curv.mean(0))
    elif curves=='raw':
        results.append(curv)
        
    if average=='perclass':
        results.append(curv.mean(1))
    elif average=='mean':
        results.append(curv.mean(1).mean())
        
    if len(results)>1:
        return tuple(results)
    elif len(results)>0:
        return results[0]