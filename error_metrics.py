import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
# from scipy.spatial.distance import dice
from scipy.spatial.distance import directed_hausdorff
import mpu.ml
import tensorflow.keras.backend as K
from model import dice_coef
def dice(vol_t,vol_s):
    """
    Computes the generalized Dice coefficient
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
    """

    vol_s= np.array(vol_s).astype(np.bool)
    vol_t= np.array(vol_t).astype(np.bool)

    if vol_s.shape != vol_t.shape:
        raise ValueError ("Shape mismatch")

    #Compute Dice coefficient
    intersection= np.logical_and(vol_s,vol_t)

    return 2. * intersection.sum() / (vol_s.sum() + vol_t.sum())
# def one_hot(a, num_classes):
#     return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
# def dice(vol_t, vol_s):
#     """
#     Dice = (2*|X & Y|)/ (|X|+ |Y|)
#          =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
#     ref: https://arxiv.org/pdf/1606.04797v1.pdf

#     """
#     a= K.constant(vol_t)
#     b=K.constant(vol_s)
#     y_true_f = K.flatten(K.one_hot(K.cast(a, 'int32'), num_classes=2)[...,0])
    
    
#     # y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(K.permute_dimensions(b,(0,2,3,1))[...,0])
#     intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
#     z= (2.0*intersection + smooth) / (K.sum(K.square(y_true_f),-1) + K.sum(K.square(y_pred_f),-1) + smooth)
#     return z
    # vol_t= np.array(vol_t).astype(np.int)
    # y_true_f= one_hot(vol_t,5)
    # # y_true_f = mpu.ml.indices2one_hot((vol_t), nb_classes=2)[...,0]
    # y_true_f_flattened= y_true_f.flatten()
    
    # y_pred_f = np.transpose(vol_s,(0,2,3,1))[...,0]
    # y_pred_f_flattened= y_pred_f.flatten()

    # intersection = np.sum(np.abs(y_true_f_flattened * y_pred_f_flattened), axis=-1)
    # return (2.0*intersection) / (np.sum(np.square(y_true_f_flattened),-1) + np.sum(np.square(y_pred_f_flattened),-1))

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
    errors['DICE']= dice(vol_t,vol_s)

  

    # errors['HAUSDORFF']= directed_hausdorff(vol_t.flatten(),vol_s.flatten())

    # non background in both
    non_bg = (vol_t != vol_t[0, 0, 0])
    errors['SSIM_NBG'] = ssim(vol_t[non_bg], vol_s[non_bg])
    dr = np.max([vol_t[non_bg].max(), vol_s[non_bg].max()]) - np.min([vol_t[non_bg].min(), vol_s[non_bg].min()])
    errors['PSNR_NBG'] = psnr(vol_t[non_bg], vol_s[non_bg], data_range=dr)
    

    vol_s_non_bg = vol_s[non_bg].flatten()
    vol_t_non_bg = vol_t[non_bg].flatten()
    errors['MSE_NBG'] = np.mean((vol_s_non_bg - vol_t_non_bg) ** 2.)
    errors['DICE_NBG'] = dice(vol_t[non_bg], vol_s[non_bg])

    return errors