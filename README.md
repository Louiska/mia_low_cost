# mia_low_cost
README_V4

Sooo. My results are not very promising (yet). I achieved a TPR@FPR=0.05 of 0.62. I tried many (many) things. Usually I would log all results (e.g., via tensorboard) and use proper config files.

I'm very confident my mistake was to use pretrained ResNets (ImageNet). I used them due to my very limited (local) computational ressources (just a CPU). I think that training a model from scratch, using an appropriate number of training epochs and a gradually degrading scheduler, like the cosine-annealing scheduler, will lead to prpper results. I've never trained a model from scratch before, therefore I thought my computational ressources might be sufficient (even with only CPUs as the training time for 20 epochs is just 20 minutes), however, with my current knowledge I would have tried to gather a (online) GPU for this task. 
 
## What did I do?

For training, I trained on the full trainingset when only a single shadow (aka. reference) model is used.
For multiple shadow models, I used a random subset of the trainset of 50% for each shadow model (with or without replacement).
-> When sampling without replacement, I further used the membership flag to ensure that 50% of the subset is (not) a member of the target model dataset.

For validation, I used the data in question to validate if the prediction of the labels is correct. I also trained on the target and trainingset when using multiple models, and using the "other" trainingset as validation. For reference samples (Z), I sampled from [the target set, the membership==1/0 set, all data]. I further normalized the input data as usually done for ResNets, as It yielded better validation results on the target (and shadow) models.

As already mentioned, I used pretrained (on ImageNet) ResNet18 models provided by PyTorch and finetuned them.

I trained for up to 20 epochs with a lr of 0.0001 (1e-4, or 4e-4 or 7e-5) and a batchsize of 64. As a scheduler, I used ReducingLROnPlateu with a factor of ~0.5 after 2 (or 4) epochs without improvement. I achieved a traing accuracy of up to 0.75 and a validation accuracy of 0.7. I always used the model performing best during training. 

Finally, within MIA.py, I simply applied the formular of RMIA. If I used a single reference model (or did not train on the targetset), I used alpha to approximate $Pr_{IN}(x)$ and $Pr_{IN}(z)$. 

## What did work well?

All models I trained were not able to achieve good values for a TPR@FPR=0.05.
Some approaches achieved a higher score than others, up to a TPR of .062. Especially high values of alpha and single shadow models whith high prediction power (accuracy) achieved my best values. But when used multiple shadow models, the TPR dropped back to approx. 0.05. 

During my tests, single shadow models always performed better than multiple shadow models.
 
## What did I try? 
- Different values of alpha
- (Different) Augmentations (used way too many augmentations at first)
- Different numbers of shadow models, from 1 up to 4
- Online & Offline mode
- Sampled z such that its (not/partially) part of the target models training set
- Train shadow models only on samples that are (not) part of the targets training set
- Train a larger (single) shadow model, resnet50 which, as the authors alreaedy mentioned, resulted in worse results as the architectures differs from the target model
- As reference samples, only choose samples with the same class as the target sample

## How I would continue
As already mentioned, I currently think that my problem is the usage of pretrained ResNets and therefore the change of the underlying trainingdataset, as it is therefore connected witht trainingdata related to ImageNet. Therefore, I would try to gather a proper GPU an train shadow-models from scratch, further: 

- I would ask for guidance/help
- Try multiquering x with majority voting
- construct a proper testset of the training subset to debug existing code, especially with regard to the usage of multiple shadow models 

Regarding good coding:
- Use Yaml or another config to properly use parameters
- Make the code more modular if I would need it any further, especially with regards to the usage of schedulers, preparation of datasets, etc.
- I would use docker 

Regarding reproducability for proper research:
- I would dive deeper into the code of the RMIA implementation, using the same training enviorment to ensure reproducability


## Questions

- Could pretraining explain my (lacking) results?
- Gerneral question: How to ensure proper reproducability and the isolated impact of the introduced method (instead of e.g., just good training parameters)?
    - Is it common to share code within a research domain, e.g., in the MIA research area?   
- Is it uncommon to use implemented methods by ML-Frameworks? E.g., Zarifzadeh et al. implemented their own scheduler and many other things