# Label-Training
## Overview
This is a scheme for training and applying a label propagation framework. It is still work in progress, so don't expect too much from it yet until it has been more extensively tested with different settings.

## Rationale
The approach assumes that segmented (into GM, WM and background) images have been aligned, so does not require the additional complexity of a convolutional approach.
The use of segmented images is to make the approach less dependent on the particular image contrasts so it generalises better to a wider variety of brain scans.
The approach assumes that there are only a relatively small number of labelled images, but many images that are unlabelled.  It therefore uses a semi-supervised learning approach, with an underlying Bayesian generative model that has relatively few weights to learn.

## Model
The approach is patch based. For each patch, a set of basis functions model both the (categorical) image to label, and the corresponding (categorical) label map.  A common set of latent variables control the two sets of basis functions, and the results are passed through a softmax so that the model encodes the means of a multinouli distribution (Böhning, 1992; Khan et al, 2010).

Continuity over patches is achieved by modelling the probability of the the latent variables within each patch conditional on the values of the latent variables in the six adjascent patches, which is a type of conditional random field (Zhang et al, 2015).  This model (with Wishart priors) gives the prior mean and covariance of a Gaussian prior over the latent variables of each patch.  Patches are updated using an iterative red-black checkerboard scheme.

## Labelling
After training, labelling a new image is relatively fast because optimising the latent variables can be formulated within a scheme similar to a recurrent Res-Net (He et al, 2016).

## References
* Böhning D. Multinomial logistic regression algorithm. Annals of the institute of Statistical Mathematics. 1992 Mar 1;44(1):197-200.
* He K, Zhang X, Ren S, Sun J. _Deep residual learning for image recognition_. In Proceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
* Khan ME, Bouchard G, Murphy KP, Marlin BM. _Variational bounds for mixed-data factor analysis_. In Advances in Neural Information Processing Systems 2010 (pp. 1108-1116).
* Zheng S, Jayasumana S, Romera-Paredes B, Vineet V, Su Z, Du D, Huang C, Torr PH. _Conditional random fields as recurrent neural networks_. In Proceedings of the IEEE international conference on computer vision 2015 (pp. 1529-1537).
