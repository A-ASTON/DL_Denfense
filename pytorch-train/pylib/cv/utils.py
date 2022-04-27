import cv2
import numpy as np

def hideXYAxis(ax):
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
def hideXYAxisTicks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
     

def isColorImg(img):
    if len(img.shape) == 2:
        return True
    if len(img.shape) == 3 and img.shape[2] == 3:
        return True

    return False
    
def matToImg(npdata):
    mmax = npdata.max()
    mmin = npdata.min()
    np_data = 255.0 * (npdata - mmin) / (mmax-mmin)
    img = np_data.astype(np.uint8)
    return img
    
def changeChannelOrder(np_data, outchannel='HWC',inchannel='CHW'):   
    if np_data.ndim == 3:
        idx = [0,1,2]
        outchannel = outchannel.upper()
        inchannel = inchannel.upper()
        for k in range(3):
            x = inchannel.index(outchannel[k])
            idx[k] = x
        img = np.transpose(np_data, tuple(idx))
        return img
    return np_data

# direction=0: horizontal, 1: vertically
def catTwoImages(img1, img2, direction=0, addline=True):
    rows1, cols1 = img1.shape[0], img1.shape[1]
    rows2, cols2 = img2.shape[0], img2.shape[1]
    
    if rows1 == rows2 and direction == 0:            
        if addline == True:
            tmpImg = np.zeros((rows1, cols1+cols2+5, 3), dtype=np.uint8)
            tmpImg[:,:,2] = 255    
            tmpImg[:,0:cols1,:] = img1
            tmpImg[:,cols1+4:-1,:] = img2
        else:
            tmpImg = np.concatenate((img1,img2),axis=1)            
        return tmpImg
        
    if cols1 == cols2 and direction == 1:
        if addline == True:
            tmpImg = np.zeros((rows1+rows2+5, cols1, 3), dtype=np.uint8)
            tmpImg[:,:,2] = 255   
            tmpImg[0:rows1,:,:] = img1
            tmpImg[rows1+4:-1,:,:] = img2
        else:
            tmpImg = np.concatenate((img1,img2),axis=0) 
            
        return tmpImg        
    