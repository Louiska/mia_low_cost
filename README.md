# mia_low_cost
ReadMe_v1

## What did I do?
As I have very limited computation resources (solely a CPU), I decided to only implement the offline, bayes approach of Zarifzadeh et al. in https://doi.org/10.48550/arXiv.2312.03262. As the direct way of approximating the Pairwise Likelihood Ratio requires a larger number of shadow models to properly determing the mean and variance of the gaussians. Hence, I also did not implement the approach by Carlini et. al. 

For training, I trained on the full trainingset when only a single shadow model is used.
When multiple shadow models have been trained, I use a random subset of the trainset of 50% for each shadow model. I further used the membership flag to ensure that 50% of the subset is [not] a member of the target model dataset.
As a validation set, I used the data in question to validate if the prediction of the labels is correct. This might lead to overfitting on the validation set which might be probelmatic especially because it's also the final testset.

I used pretrained (ImageNet) ResNet18 models provided by PyTorch and finetuned them as shadow models.

## What did work well?
All models I trained were not able to achieve good values for a TPR@FPR=0.05.
Some approaches achieved a higher score than others, up to a TPR of .062, but when highered the number of shadow models, the value dropped back to approx. 0.05.

I guess this has the following reasons:
- The (training) dataset(s) is/are fairly small (20000 samples), the shadow models easily overfit on the trainset. 
- The accuracy of the target model as well as the shadow models is fairly small on the targetset; both ~0.7
    - The model is fairly small by itself (ResNet18)

Single shadow models always performed better than multiple shadow models (and I'm very curious why that's the case)
I guess it's because my predictions are mostly random - and having multiple models evens out randomness, which might be able to predict members to a decent degree.
 
## What did I try?
- Different values of alpha
- (Different) Augmentations
- Different numbers of shadow models, from 1 up to 4
- (In and Out shadow model regarding z)I tried to draw Z such that its not in the training population of the target set. Additionally, I trained two shadow models, that trained both of the complementary half of the training set, such that half the model is trained on the one half, the other on the other half of the train dataset. When sampling z \nin D, one shadow model is trained, the other one is not trained on z
- Calculating Pr(z) the same way Pr(x) is calculating, i.e., balancing the prediction via alpha by approximating the inner prediction
- sample z such that its not part of the training set of the target model. Thne debugged everything step by step (i.e., a LOT)
## How I would continue
- First of all in the scope of this task, I would ask for help/guidance.

- I would construct a proper testset by e.g., fetching more samples, to debug existing code, especially with regard to the behavior of different alpha values and the influence of the number of shadow models
- If I would have the computational ressources, 
    - try online mode
    - I would use larger models to debug if my results are lacking due to low prediction power (accuracy)
    - Instead of calculating the bayes approach, I would try to approximate the direct approach using gaussians with an appropriate amount of shadow models as its promised to yield slightly better results
- I did not yet analyse the dataset itself and hence its lacking a deeper analysis of the samples, which might be relevant especially in regard to augmentations. 