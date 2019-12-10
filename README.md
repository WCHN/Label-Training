# Label-Training

## Overview
In addition to the vesion https://github.com/WCHN/Label-Training, this new version including some new features:
1.weighting
The displacement of certain amount of voxel for one voxel along each direction is controlled by is a data augmentation method we used in this work. We weight the contribution of each row of the data according to the amount of jitter. If jitter is zero in each direction, then it will be weighted more heavily than if it is displaced by 1 voxel along all three dimensions. This weighting is based on Gaussian probability with different values of standard deviation. 
2. less computation cost
The model is saved in a directory, with a different file to encode information for each slice. Files would be loaded and data is progressively cleared as required.  
3. label map reconstruction and warpping 
Reconstrct the label map from the binary segmented data and warping the predicted label probabilities (which are all nonlinearly aligned with each other) back to match the original image volumes, and calcualte the Dice coefficient with the ground truth.
## Usage
run the main_savememory.m to get the trained model, and then run the labelmaprecon.m to get the reconstructed label map, the reconstructed labelmap can be warpped back to the original space by using test_job.m. code works on both 2D and 3D T1 MR images where some changes need to be made accordingly.
PS: the latest SPM 12 is required to run this model and is available to download from https://www.fil.ion.ucl.ac.uk/spm/software/spm12/




