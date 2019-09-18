import numpy as np 
import nibabel as nib 
import os

data = [nib.load('.' + '/' + f).get_data() for f in np.sort(os.listdir('./'))]
name = ['bet-ms0056-05.nii.gz','bet-ms0060-01.nii.gz',
'bet-ms0064-04.nii.gz','bet-ms0069-04.nii.gz','bet-ms0070-04.nii.gz','bet-ms0071-04.nii.gz',
'bet-ms0073-06.nii.gz','bet-ms0077-04.nii.gz','bet-ms0078-05.nii.gz','bet-ms0086-04.nii.gz',
'bet-ms0135-04.nii.gz','bet-ms0165-03.nii.gz','bet-ms0170-05.nii.gz','bet-ms0171-05.nii.gz',]
for i in range(0,14):
  y=np.squeeze(data[i],3)
  img = nib.Nifti1Image(y, np.eye(4))
  nib.save(img, name[i])