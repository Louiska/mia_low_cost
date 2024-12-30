# mia_low_cost
README_V3

Sooo. Somehow my results are not very promising (yet). I achieved a TPR@FPR=0.05 of 0.62. I tried many (many) things. Usually I would log all results and use proper config files, but I did not thought it would be such a hustle for me to achieve proper results. Hence, the current state of the code is just the last approach I tried.


## What did I do?
As I have very limited computational resources (only a CPU), I decided to mainly implement the offline, RMIA-Bayes approach of Zarifzadeh et al. in https://doi.org/10.48550/arXiv.2312.03262. As the direct way of approximating the Pairwise Likelihood Ratio requires a larger number of shadow models to properly determing the mean and variance of the Gaussians. Therefore, I also did not implement the approach by Carlini et. al. 

For training, I trained on the full trainingset when only a single shadow (aka. reference) model is used.
For multiple shadow models, I used a random subset of the trainset of 50% for each shadow model. I further used the membership flag to ensure that 50% of the subset is (not) a member of the target model dataset.
-> I tried training the models with and without replacement of the training set, and also to do some sort of (non replacement) cross-training, leveraging both given datasets by ensuring that 50% of the member-data is included in each trainingset.

As a validation set, I used the data in question to validate if the prediction of the labels is correct (or in the cross scenario, the "other" dataset). This might lead to overfitting on the validation set, but this has also been done by the authors. For reference samples (z), I only used samples that [are,not] part of the targets training set [or just sampled of both datasets]. I used Augmentations to prevent overfitting on the training set. I further normalized the input data as usually done for ResNets, as It yielded better validation results on the target (and shadow) models.

I used pretrained (on ImageNet) ResNet18 models provided by PyTorch and finetuned them.

I trained for up to 20 epochs with a lr of 0.0001 (1e-4, or 4e-4 or 7e-5), batchsize of 64. As a scheduler, I used ReducingLROnPlateu with a factor of ~0.5 after 2 (or 4) epochs without improvement. I achieved a traing accuracy of up to 0.75 and a validation accuracy of 0.7. Maybe I used too many augmentations during training, but otherwise I had problems with overfitting. I always used the model performing best during training, which however was during the last epochs anyway. 

## What did work well?

All models I trained were not able to achieve good values for a TPR@FPR=0.05.
Some approaches achieved a higher score than others, up to a TPR of .062. Especially high values of alpha and single shadow models whith high prediction power (accuracy) achieved my best values. But when used multiple shadow models, the TPR dropped back to approx. 0.05. 

I guess for a single shadow model this has the following reason: 
- The accuracy of the target model as well as the shadow models is fairly small on the targetset; both ~0.7

During my tests, single shadow models always performed better than multiple shadow models (and I'm very curious why that's the case).

Might one of the following actions be one of the reasons for this?
- Multiple models are trained by using a subset of the trainset, e.g., 10k Samples instead of 20k. I did not train on the targetset. Some Classes have very low data (up to 20 entries), splitting the data probably leads to overfitting or no training regarding this classes. 
- I always used pretrained resnet18 for reference models
- I assumed its better if z is not a member of the target models training set, but has been trained on by half the reference models to properly calculate their in/out prediction and average on the final result. However, the predictions vary a lot when multiple reference models are trained (their variance is very low) vs when a single reference model is trained.  
 
## What did I try?
- Different values of alpha
- (Different) Augmentations
- Different numbers of shadow models, from 1 up to 4
- Online & Offline mode
- Sampled z such that its (not/partially) part of the target models training set
- Train shadow models only on samples that are (not) part of the targets training set
- Train a larger (single) shadow model, resnet50 which, as the authors alreaedy mentioned, resulted in worse results as the architectures are not the same
- As reference samples, only choose samples with the same class as the target sample

## How I would continue
First of all in the scope of this task, I would ask for help/guidance. Especially in regard to why multiple shadow models yield worse results. 

- Try multiquering x with majority voting
- Further, I would construct a proper testset of the training subset to debug existing code, especially with regard to the usage of multiple shadow models 