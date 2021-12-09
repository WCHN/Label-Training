# Factorisation-based Image Labelling
## Overview
This is a scheme for training and applying the factorisation-based image labelling (FIL) framework. Some functionality from SPM12 is required for handling images (available from https://www.fil.ion.ucl.ac.uk/spm/software/spm12/). It is still work in progress, so don't expect too much from it until it has been properly debugged and refactored, as well as more extensively tested with different settings.

## Rationale
The approach assumes that segmented (into GM, WM and background) images have been aligned, so does not require the additional complexity of a convolutional approach.
The use of segmented images is to make the approach less dependent on the particular image contrasts so it generalises better to a wider variety of brain scans.
The approach assumes that there are only a relatively small number of labelled images, but many images that are unlabelled.  It therefore uses a semi-supervised learning approach, with an underlying Bayesian generative model that has relatively few weights to learn.

## Model
The approach is patch based. For each patch, a set of basis functions model both the (categorical) image to label, and the corresponding (categorical) label map.  A common set of latent variables control the two sets of basis functions, and the results are passed through a softmax so that the model encodes the means of a multinouli distribution (Böhning, 1992; Khan et al, 2010).

Continuity over patches is achieved by modelling the probability of the latent variables within each patch conditional on the values of the latent variables in the six adjacent patches, which is a type of conditional random field (Zhang et al, 2015; Brudfors et al, 2019).  This model (with Wishart priors) gives the prior mean and covariance of a Gaussian prior over the latent variables of each patch.  Patches are updated using an iterative red-black checkerboard scheme.

## Labelling
After training, labelling a new image is relatively fast because optimising the latent variables can be formulated within a scheme similar to a recurrent Res-Net (He et al, 2016).

## Example Code
`Example_FIL_train.m` is an example training script and `fil_label.m` provides an example of how to apply a trained model.  Note that spm12 (https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) and the Multi-Brain toolbox (https://github.com/WTCN-computational-anatomy-group/mb) are required to run the code. This toolbox, along with the Multi-Brain toolbox, will be included in a future SPM release.

## References
* Böhning D. _Multinomial logistic regression algorithm_. Annals of the institute of Statistical Mathematics. 1992 Mar 1;44(1):197-200.
* Brudfors M, Balbastre Y & Ashburner J. Nonlinear Markov Random Fields Learned via Backpropagation. Accepted for 26th international conference on Information Processing in Medical Imaging (IPMI 2019). Preprint available from http://arxiv.org/abs/1902.10747 .
* He K, Zhang X, Ren S, Sun J. _Deep residual learning for image recognition_. In Proceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
* Khan ME, Bouchard G, Murphy KP, Marlin BM. _Variational bounds for mixed-data factor analysis_. In Advances in Neural Information Processing Systems 2010 (pp. 1108-1116).
* Yan Y, Balbastre Y, Brudfors M, Ashburner J. _Factorisation-based Image Labelling_. arXiv preprint arXiv:2111.10326. 2021 Nov 19.
* Zheng S, Jayasumana S, Romera-Paredes B, Vineet V, Su Z, Du D, Huang C, Torr PH. _Conditional random fields as recurrent neural networks_. In Proceedings of the IEEE international conference on computer vision 2015 (pp. 1529-1537).

## Acknowledgements
This work was funded by the EU Human Brain Project’s Grant Agreement No 785907 (SGA2).

