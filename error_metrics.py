import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
# from scipy.spatial.distance import dice
from scipy.spatial.distance import directed_hausdorff
import mpu.ml
import tensorflow.keras.backend as K



# def computeDice(vol_s, vol_t):
    
#     vol_s= vol_s>0.5

#     pred = np.ndarray.flatten(np.clip(vol_s,0,1))
#     gt = np.ndarray.flatten(np.clip(vol_t,0,1))
#     intersection = np.sum(pred * gt) 
#     union = np.sum(pred) + np.sum(gt)   
#     return np.round((2 * intersection)/(union),decimals=5)

# def computeDice(vol_s, vol_t):
#     """ Returns
#     -------
#     DiceArray : floats array
          
#           Dice coefficient as a float on range [0,1].
#           Maximum similarity = 1
#           No similarity = 0 """
          
#     n_classes = int( np.max(vol_t) + 1)
   
#     DiceArray = []
    
    
#     for c_i in xrange(1,n_classes):
#         idx_Auto = np.where(vol_s.flatten() == c_i)[0]
#         idx_GT   = np.where(vol_t.flatten() == c_i)[0]
        
#         autoArray = np.zeros(vol_s.size,dtype=np.bool)
#         autoArray[idx_Auto] = 1
        
#         gtArray = np.zeros(vol_s.size,dtype=np.bool)
#         gtArray[idx_GT] = 1
        
#         dsc = dice(autoArray, gtArray)
        
#         #dice = np.sum(autoSeg[groundTruth==c_i])*2.0 / (np.sum(autoSeg) + np.sum(groundTruth))
#         DiceArray.append(dsc)
        
#     return DiceArray


def dice(vol_s,vol_t):
#     """
    
#     im1 : array-like, bool
#         Any array of arbitrary size. If not boolean, will be converted.
#     im2 : array-like, bool
#         Any other array of identical size. If not boolean, will be converted.
#     Returns
#     -------
#     dice : float
#     Dice coefficient as a float on range [0,1].
#         Maximum similarity = 1
#         No similarity = 0
#     """
    
    smooth = 1
    pred = np.ndarray.flatten(np.clip(vol_s,0,1))
    gt = np.ndarray.flatten(np.clip(vol_t,0,1))
    intersection = np.sum(pred * gt) 
    union = np.sum(pred) + np.sum(gt)   
    return np.round((2 * intersection + smooth)/(union + smooth),decimals=5)

    


def ErrorMetrics(vol_s, vol_t):
    # calculate various error metrics.
    # vol_s should be the synthesized volume (a 3d numpy array) or an array of these volumes
    # vol_t should be the ground truth volume (a 3d numpy array) or an array of these volumes

    vol_s = np.squeeze(vol_s)
    vol_t = np.squeeze(vol_t)

    assert len(vol_s.shape) == len(vol_t.shape) == 3
    assert vol_s.shape[0] == vol_t.shape[0]
    assert vol_s.shape[1] == vol_t.shape[1]
    assert vol_s.shape[2] == vol_t.shape[2]

    vol_s[vol_t == 0] = 0
    vol_s[vol_s < 0] = 0



    errors = {}

    errors['MSE'] = np.mean((vol_s - vol_t) ** 2.)
    errors['SSIM'] = ssim(vol_t, vol_s)
    dr = np.max([vol_s.max(), vol_t.max()]) - np.min([vol_s.min(), vol_t.min()])
    errors['PSNR'] = psnr(vol_t, vol_s, data_range=dr)
    errors['DICE']= dice(vol_s,vol_t)

  

    # errors['HAUSDORFF']= directed_hausdorff(vol_t.flatten(),vol_s.flatten())

    # non background in both
    non_bg = (vol_t != vol_t[0, 0, 0])
    errors['SSIM_NBG'] = ssim(vol_t[non_bg], vol_s[non_bg])
    dr = np.max([vol_t[non_bg].max(), vol_s[non_bg].max()]) - np.min([vol_t[non_bg].min(), vol_s[non_bg].min()])
    errors['PSNR_NBG'] = psnr(vol_t[non_bg], vol_s[non_bg], data_range=dr)
    

    vol_s_non_bg = vol_s[non_bg].flatten()
    vol_t_non_bg = vol_t[non_bg].flatten()
    errors['MSE_NBG'] = np.mean((vol_s_non_bg - vol_t_non_bg) ** 2.)
    errors['DICE_NBG'] = dice(vol_s[non_bg], vol_t[non_bg])

    return errors