# **The Algonauts Project 2023 Challenge - devkit tutorial**

The quest to understand the nature of human intelligence and engineer more advanced forms of artificial intelligence are becoming increasingly intertwined. The Algonauts Project brings biological and machine intelligence researchers together on a common platform to exchange ideas and advance both fields. The 2023 challenge focuses on explaining neural visual responses to complex naturalistic scenes.

The challenge is based on the [Natural Scenes Dataset][nsd] (NSD), a massive dataset of 7T fMRI responses to images of natural scenes coming from the [COCO dataset][coco]. The goal of the challenge is to promote the development of cutting-edge encoding models which accurately predict (i.e., encode) the fMRI responses to visual input. You can read more about the challenge in the [website][web] and [paper][paper].

Here we provide the development kit tutorial of the challenge where you will learn how to:
* Load and visualize the challenge fMRI data, functionally different areas of the visual cortex called regions-of-interest (ROIs), and the stimulus images.
* Build [linearizing encoding models][encoding] using a pretrained [AlexNet][alexnet] architecture, evaluate them and visualize the resulting prediction accuracy (i.e., encoding accuracy).
* Prepare the predicted brain responses to the test images in the right format for submission to the challenge leaderboard.

Please watch [this video][!!!!!!!!!!!!! link !!!!!!!!!!!!!] for an introduction to the Algonauts Project 2023 Challenge and a walkthrough of this development kit tutorial. You can also run this tutorial on [Google Colab][colab].

Should you experience problems with the code, please get in touch with Ale (alessandro.gifford@gmail.com).

[nsd]: https://doi.org/10.1038/s41593-021-00962-x
[coco]: https://cocodataset.org/#home
[web]: http://algonauts.csail.mit.edu
[paper]: https://arxiv.org/abs/2301.03198
[encoding]: https://www.sciencedirect.com/science/article/pii/S1053811910010657
[alexnet]: https://arxiv.org/abs/1404.5997
[colab]: https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link
