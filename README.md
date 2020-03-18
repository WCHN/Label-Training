# Label-Training

## Overview
In addition to the vesion https://github.com/WCHN/Label-Training, this version including some new features:

**Weighting**
We weight the contribution of each row of the data according to the amount of jitter. If jitter is zero in each direction, then it will be weighted more heavily than if it is displaced by 1 voxel along all three dimensions. This weighting is based on Gaussian probability with different values of standard deviation. 

**Less computation cost**
The model is saved in a directory, with a different file to encode information for each slice. Files would be loaded and data is progressively cleared as required.  

**Label map reconstruction and warpping**
Reconstrct the label map from the binary segmented data and warping the predicted label probabilities (which are all nonlinearly aligned with each other) back to match the original image volumes, and calcualte the Dice coefficient with the ground truth.

## Usage
Generate the warpped label first using  **labelwarp.m** and run **main_savememory.m** to get the predicted label map from the trained model, and the dice score is computed to show the results of the segmentation. Code works on both 2D and 3D T1 MR images where some hyperparameters changes need to be made accordingly.

PS: The latest SPM 12 is required to run this model and is available to download from https://www.fil.ion.ucl.ac.uk/spm/software/spm12/




