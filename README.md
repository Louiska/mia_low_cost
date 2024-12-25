# mia_low_cost
README_V2

## What did I do?
As I have very limited computational resources (solely a CPU), I decided to only implement the offline, bayes approach of Zarifzadeh et al. in https://doi.org/10.48550/arXiv.2312.03262. As the direct way of approximating the Pairwise Likelihood Ratio requires a larger number of shadow models to properly determing the mean and variance of the gaussians. I also did not implement the approach by Carlini et. al. 

For training, I trained on the full trainingset when only a single shadow (aka. reference) model is used.
For multiple shadow models, I used a random subset of the trainset of 50% for each shadow model. I further used the membership flag to ensure that 50% of the subset is (not) a member of the target model dataset.
As a validation set, I used the data in question to validate if the prediction of the labels is correct. This might lead to overfitting on the validation set, but this has also been done by the authors. For reference samples (z), I only used samples that are (not/partially) part of the targets training set.

I used pretrained (on ImageNet) ResNet18 models provided by PyTorch and finetuned them as shadow models.

## What did work well?

All models I trained were not able to achieve good values for a TPR@FPR=0.05.
Some approaches achieved a higher score than others, up to a TPR of .062, but when used multiple shadow models, the value dropped back to approx. 0.05.

I guess this has the following reasons: 
- The accuracy of the target model as well as the shadow models is fairly small on the targetset; both ~0.7
    - The model is fairly small by itself (ResNet18)

Single shadow models always performed better than multiple shadow models (and I'm very curious why that's the case)
 
## What did I try?
- Different values of alpha
- (Different) Augmentations
- Different numbers of shadow models, from 1 up to 4
- Sampled z such that its (not/partially) part of the target models training set
- Train shadow models only on samples that are (not) part of the targets training set

## How I would continue
- First of all in the scope of this task, I would ask for help/guidance. Especially in regard to why multiple shadow models yield worse results. 

- Further, I would construct a proper testset of the training subset to debug existing code, especially with regard to the behavior of different alpha values and the influence of the number of shadow models 
- If I would have the computational ressources, 
    - try online mode
    - I would use larger models to debug if my results are lacking due to low prediction power (accuracy)
    - Instead of calculating the bayes approach, I would try to approximate the direct approach using gaussians with an appropriate amount of shadow models as its promised to yield slightly better results