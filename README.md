

1. Make sure the buffer never contains samples that
not from a past task (monitor the data_valuation part)
2. Should give more priority to training on a new task.
Local training:
    Loop through data in this task
        sample from buffer from old tasks
        concat data
        train

Sharing
Data Valuation & receive
    each shared data task is already in the buffer and the data valuation is overwhelmingly one-sided.
        - add all data to buffer
        - solve by backtesting again...

Running into OOM issue with this dataset...



CIFAR-10
VGG-16 can get 79% accuracy on the data (Takes 6 mins to train per agent) while our faster model can get about 60%.


There are a lot more datasets such as the car or celebA or even omiglot that might be more useful for data pooling!

The key is that each class performance valuation should be stable and discriminative!
(regardless of class imbalance)


Clustering seems to be very difficult for cifar-10. Maybe vgg16 will stabilize the data
valuation metric directly using the improvement output.

Maybe look at t=2 instead of t=1... so that the accuracy is not as high!

MNIST similarity:

https://www.kaggle.com/code/kmader/image-similarity-with-siamese-networks


TODO:
Figure out the cifar-10 stuff...
https://github.com/timsainb/ParametricUMAP_paper/blob/master/notebooks/semisupervised/plot-latent-projections/plot-last-layer-embeddings.ipynb
Has to use semi-supervised learning (labels) to do the embedding correctly!!

NLP tasks

https://howardhsu.github.io/doc/prelim_slides.pdf




# MNIST and FashionMNIST experiments

1. Establish that more data is better!
2. Establish that there is "bad" and "good" data. (introducing a mixed of MNIST and FashionMNIST might be too aggressive!)
file:///Users/longle/Downloads/LifelongTeacherStudent.pdf
Introduce bad and good data by only including certain variations of the data!

Once we introduce fixed-memory buffer, we need to do resovoir sampling and modify the clustering metric preference score to take into account
that we need to get more data that we don't currently have!


One-to-all: is not going to work for any large enough N.



TODO:
1. switch stopping criterion from val to test set. [x]
2. Visualize the test / train / val as you train it. [x]
3. Run a class-balanced and class-imbalanced experiments on MNIST and see the confusion matrix. [x]
4. Implement dedup!



# Dedup
it's more efficient to do dedup from the sender side (on the indices), but in general dedup should be actually implemented in the buffer to allow for circumstances like finite-sized buffer.

NOTE: there is no easy way to support dedup data so in the get candidate data we will only consider the current task, which will eliminate this dedup need but also makes it not possible to have shifting preference (second thoughts about a data cls in the past)

TODO: incorporate dedup into the rewards functions as well!


Cifar-10 is really biazare!!

## Ablation study
- (0, 1); (2, 3) and (0, 5); (1, 6) to demonstrate
    - dedup
    - shifting preference
    - performance should capture all of these.
- why performance-based >> heuristic class-based. 
Because class-based does not know which task should be prioritized. Construct a scenario between an easy and a hard "task" and see.




should probably switch to task incremental learning instead of class incremental...
(to avoid intraclass/intratask transfer)
Also, what would be the architecture for these shits...


# NOTE
Debugging and monitoring loss to make sure that these stuff are trained
properly is important!

For cifar10 and mnist, batch-size should be pretty small (32) to avoid training loss not going down issue!


Do the image similarity stuff.

# Bandit
Mutli-armed and contextual bandit


# Plotting

https://gdmarmerola.github.io/non-stationary-bandits/

This dude plot is awesome!
https://github.com/gdmarmerola?tab=repositories


https://github.com/gdmarmerola/advanced-bandit-problems/blob/master/notebooks/approximate_bayes_for_bandits.ipynb



## DEBUG: switching from val to test fucks this experiment up...


__Validation__


python experiments/two_staged.py
Global seed set to 0
DEBUG    dataset dim: torch.Size([1, 28, 28])                                                                                                                                                                                                          
DEBUG    subsets length: [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]                                                                                                                                                                       
DEBUG    dataset dim: torch.Size([1, 28, 28])                                                                                                                                                                                                          
DEBUG    subsets length: [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]                                                                                                                                                                  
size: 64
train_size: 64, num_cls_per_task: 4
DEBUG    DATASET: mnist, train_size: 64, val_size: 64, test_size: 892                                                                                                                                                                                  
DEBUG    subsets length: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64] with provided size 64                                                                                                                                                                
DEBUG    subsets length: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64] with provided size 64                                                                                                                                                                
DEBUG    subsets length: [892, 892, 892, 892, 892, 892, 892, 892, 892, 892] with provided size 892                                                                                                                                                     
train_size: 64, num_cls_per_task: 4
DEBUG    DATASET: mnist, train_size: 64, val_size: 64, test_size: 892                                                                                                                                                                                  
DEBUG    subsets length: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64] with provided size 64                                                                                                                                                                
DEBUG    subsets length: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64] with provided size 64                                                                                                                                                                
DEBUG    subsets length: [892, 892, 892, 892, 892, 892, 892, 892, 892, 892] with provided size 892                                                                                                                                                     
INFO     MNIST num parameters: 13998                                                                                                                                                                                                                   
INFO     MNIST num parameters: 13998                                                                                                                                                                                                                   
INFO     epoch: 1, step 0 loss: 0.000 | val_loss 2.289 | train_acc 0.027 | val_acc 0.023 | test_acc 0.029 | past_task_test_acc 0.029                                                                                                                   
INFO     epoch: 6, step 20 loss: 101.974 | val_loss 1.576 | train_acc 0.250 | val_acc 0.250 | test_acc 0.250 | past_task_test_acc 0.250                                                                                                                
INFO     epoch: 11, step 40 loss: 72.681 | val_loss 1.062 | train_acc 0.750 | val_acc 0.770 | test_acc 0.768 | past_task_test_acc 0.768                                                                                                                
INFO     epoch: 16, step 60 loss: 42.746 | val_loss 0.540 | train_acc 0.848 | val_acc 0.867 | test_acc 0.839 | past_task_test_acc 0.839                                                                                                                
INFO     epoch: 21, step 80 loss: 20.470 | val_loss 0.317 | train_acc 0.859 | val_acc 0.910 | test_acc 0.881 | past_task_test_acc 0.881                                                                                                                
INFO     epoch: 26, step 100 loss: 14.878 | val_loss 0.293 | train_acc 0.887 | val_acc 0.891 | test_acc 0.886 | past_task_test_acc 0.886                                                                                                               
INFO     epoch: 31, step 120 loss: 11.233 | val_loss 0.225 | train_acc 0.945 | val_acc 0.922 | test_acc 0.932 | past_task_test_acc 0.932                                                                                                               
INFO     epoch: 36, step 140 loss: 7.171 | val_loss 0.222 | train_acc 0.938 | val_acc 0.918 | test_acc 0.928 | past_task_test_acc 0.928                                                                                                                
INFO     epoch: 41, step 160 loss: 10.593 | val_loss 0.242 | train_acc 0.949 | val_acc 0.926 | test_acc 0.935 | past_task_test_acc 0.935                                                                                                               
INFO     epoch: 46, step 180 loss: 16.529 | val_loss 0.312 | train_acc 0.922 | val_acc 0.883 | test_acc 0.886 | past_task_test_acc 0.886                                                                                                               
INFO     epoch: 51, step 200 loss: 10.396 | val_loss 0.236 | train_acc 0.961 | val_acc 0.914 | test_acc 0.930 | past_task_test_acc 0.930                                                                                                               
INFO     epoch: 56, step 220 loss: 8.781 | val_loss 0.240 | train_acc 0.957 | val_acc 0.910 | test_acc 0.928 | past_task_test_acc 0.928                                                                                                                
INFO     epoch: 61, step 240 loss: 5.292 | val_loss 0.243 | train_acc 0.969 | val_acc 0.934 | test_acc 0.938 | past_task_test_acc 0.938                                                                                                                
INFO     epoch: 66, step 260 loss: 5.979 | val_loss 0.248 | train_acc 0.969 | val_acc 0.922 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                
INFO     epoch: 71, step 280 loss: 10.377 | val_loss 0.244 | train_acc 0.973 | val_acc 0.918 | test_acc 0.937 | past_task_test_acc 0.937                                                                                                               
INFO     epoch: 76, step 300 loss: 10.142 | val_loss 0.263 | train_acc 0.980 | val_acc 0.922 | test_acc 0.936 | past_task_test_acc 0.936                                                                                                               
INFO     epoch: 81, step 320 loss: 3.832 | val_loss 0.264 | train_acc 0.980 | val_acc 0.922 | test_acc 0.939 | past_task_test_acc 0.939                                                                                                                
INFO     epoch: 86, step 340 loss: 4.929 | val_loss 0.321 | train_acc 0.961 | val_acc 0.906 | test_acc 0.919 | past_task_test_acc 0.919                                                                                                                
INFO     epoch: 91, step 360 loss: 2.182 | val_loss 0.270 | train_acc 0.984 | val_acc 0.922 | test_acc 0.939 | past_task_test_acc 0.939                                                                                                                
INFO     epoch: 96, step 380 loss: 2.739 | val_loss 0.280 | train_acc 0.980 | val_acc 0.934 | test_acc 0.938 | past_task_test_acc 0.938                                                                                                                
INFO     epoch: 101, step 400 loss: 5.069 | val_loss 0.280 | train_acc 0.988 | val_acc 0.926 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                               
INFO     epoch: 106, step 420 loss: 5.958 | val_loss 0.288 | train_acc 0.984 | val_acc 0.918 | test_acc 0.937 | past_task_test_acc 0.937                                                                                                               
INFO     epoch: 111, step 440 loss: 3.084 | val_loss 0.312 | train_acc 0.984 | val_acc 0.922 | test_acc 0.937 | past_task_test_acc 0.937                                                                                                               
INFO     Early stopping at epoch 111 with best val loss 0.066                                                                                                                                                                                          
WARNING  Randomly routing...                                                                                                                                                                                                                           
WARNING  Router available dataset size: 256                                                                                                                                                                                                            
DEBUG    No. of duplicates: 2                                                                                                                                                                                                                          
CRITICAL Buffer size: 318 cls distribution (tensor([0, 3, 4, 9]), tensor([126,  64,  64,  64]))                                                                                                                                                        
INFO     epoch: 1, step 5 loss: 5.673 | val_loss 0.231 | train_acc 0.973 | val_acc 0.922 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                   
INFO     epoch: 6, step 30 loss: 0.925 | val_loss 0.246 | train_acc 0.973 | val_acc 0.922 | test_acc 0.939 | past_task_test_acc 0.939                                                                                                                  
INFO     epoch: 11, step 55 loss: 2.150 | val_loss 0.244 | train_acc 0.965 | val_acc 0.922 | test_acc 0.936 | past_task_test_acc 0.936                                                                                                                 
INFO     epoch: 16, step 80 loss: 0.999 | val_loss 0.250 | train_acc 0.969 | val_acc 0.926 | test_acc 0.939 | past_task_test_acc 0.939                                                                                                                 
INFO     epoch: 21, step 105 loss: 1.299 | val_loss 0.289 | train_acc 0.957 | val_acc 0.918 | test_acc 0.925 | past_task_test_acc 0.925                                                                                                                
INFO     epoch: 26, step 130 loss: 5.446 | val_loss 0.282 | train_acc 0.973 | val_acc 0.918 | test_acc 0.936 | past_task_test_acc 0.936                                                                                                                
INFO     epoch: 31, step 155 loss: 2.277 | val_loss 0.295 | train_acc 0.977 | val_acc 0.918 | test_acc 0.933 | past_task_test_acc 0.933                                                                                                                
INFO     epoch: 36, step 180 loss: 0.784 | val_loss 0.316 | train_acc 0.973 | val_acc 0.922 | test_acc 0.933 | past_task_test_acc 0.933                                                                                                                
INFO     epoch: 41, step 205 loss: 0.610 | val_loss 0.306 | train_acc 0.980 | val_acc 0.918 | test_acc 0.930 | past_task_test_acc 0.930                                                                                                                
INFO     epoch: 46, step 230 loss: 4.646 | val_loss 0.318 | train_acc 0.980 | val_acc 0.910 | test_acc 0.930 | past_task_test_acc 0.930                                                                                                                
INFO     epoch: 51, step 255 loss: 0.277 | val_loss 0.323 | train_acc 0.977 | val_acc 0.914 | test_acc 0.934 | past_task_test_acc 0.934                                                                                                                
INFO     epoch: 56, step 280 loss: 2.799 | val_loss 0.343 | train_acc 0.977 | val_acc 0.918 | test_acc 0.930 | past_task_test_acc 0.930                                                                                                                
INFO     epoch: 61, step 305 loss: 0.215 | val_loss 0.335 | train_acc 0.984 | val_acc 0.910 | test_acc 0.938 | past_task_test_acc 0.938                                                                                                                
INFO     epoch: 66, step 330 loss: 0.594 | val_loss 0.354 | train_acc 0.984 | val_acc 0.922 | test_acc 0.938 | past_task_test_acc 0.938                                                                                                                
INFO     Early stopping at epoch 66 with best val loss 0.074                                                                                                                                                                                           
CRITICAL Class 0 before 0.93359375 after 0.92578125 len 64 contribution: -0.008368200836820083                                                                                                                                                         
DEBUG    No. of duplicates: 1                                                                                                                                                                                                                          
CRITICAL Buffer size: 319 cls distribution (tensor([0, 3, 4, 9]), tensor([ 64, 127,  64,  64]))                                                                                                                                                        
INFO     epoch: 1, step 5 loss: 2.586 | val_loss 0.242 | train_acc 0.965 | val_acc 0.914 | test_acc 0.937 | past_task_test_acc 0.937                                                                                                                   
INFO     epoch: 6, step 30 loss: 3.969 | val_loss 0.227 | train_acc 0.977 | val_acc 0.922 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                  
INFO     epoch: 11, step 55 loss: 4.186 | val_loss 0.220 | train_acc 0.965 | val_acc 0.930 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                 
INFO     epoch: 16, step 80 loss: 3.082 | val_loss 0.230 | train_acc 0.973 | val_acc 0.922 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                 
INFO     epoch: 21, step 105 loss: 1.025 | val_loss 0.254 | train_acc 0.980 | val_acc 0.914 | test_acc 0.928 | past_task_test_acc 0.928                                                                                                                
INFO     epoch: 26, step 130 loss: 1.246 | val_loss 0.328 | train_acc 0.973 | val_acc 0.902 | test_acc 0.922 | past_task_test_acc 0.922                                                                                                                
INFO     epoch: 31, step 155 loss: 1.023 | val_loss 0.269 | train_acc 0.973 | val_acc 0.910 | test_acc 0.934 | past_task_test_acc 0.934                                                                                                                
INFO     epoch: 36, step 180 loss: 0.687 | val_loss 0.285 | train_acc 0.980 | val_acc 0.914 | test_acc 0.926 | past_task_test_acc 0.926                                                                                                                
INFO     epoch: 41, step 205 loss: 1.672 | val_loss 0.293 | train_acc 0.973 | val_acc 0.910 | test_acc 0.923 | past_task_test_acc 0.923                                                                                                                
INFO     epoch: 46, step 230 loss: 3.875 | val_loss 0.293 | train_acc 0.984 | val_acc 0.910 | test_acc 0.930 | past_task_test_acc 0.930                                                                                                                
INFO     epoch: 51, step 255 loss: 1.628 | val_loss 0.325 | train_acc 0.977 | val_acc 0.898 | test_acc 0.925 | past_task_test_acc 0.925                                                                                                                
INFO     epoch: 56, step 280 loss: 0.586 | val_loss 0.317 | train_acc 0.980 | val_acc 0.910 | test_acc 0.930 | past_task_test_acc 0.930                                                                                                                
INFO     epoch: 61, step 305 loss: 0.700 | val_loss 0.352 | train_acc 0.977 | val_acc 0.902 | test_acc 0.924 | past_task_test_acc 0.924                                                                                                                
INFO     Early stopping at epoch 61 with best val loss 0.070                                                                                                                                                                                           
CRITICAL Class 3 before 0.93359375 after 0.9296875 len 64 contribution: -0.0041841004184100415                                                                                                                                                         
DEBUG    No. of duplicates: 0                                                                                                                                                                                                                          
CRITICAL Buffer size: 320 cls distribution (tensor([0, 3, 4, 9]), tensor([ 64,  64, 128,  64]))                                                                                                                                                        
INFO     epoch: 1, step 5 loss: 2.402 | val_loss 0.241 | train_acc 0.965 | val_acc 0.922 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                   
INFO     epoch: 6, step 30 loss: 1.069 | val_loss 0.233 | train_acc 0.973 | val_acc 0.926 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                  
INFO     epoch: 11, step 55 loss: 4.224 | val_loss 0.255 | train_acc 0.973 | val_acc 0.922 | test_acc 0.945 | past_task_test_acc 0.945                                                                                                                 
INFO     epoch: 16, step 80 loss: 6.621 | val_loss 0.369 | train_acc 0.918 | val_acc 0.891 | test_acc 0.891 | past_task_test_acc 0.891                                                                                                                 
INFO     epoch: 21, step 105 loss: 2.450 | val_loss 0.297 | train_acc 0.973 | val_acc 0.918 | test_acc 0.926 | past_task_test_acc 0.926                                                                                                                
INFO     epoch: 26, step 130 loss: 2.950 | val_loss 0.282 | train_acc 0.973 | val_acc 0.918 | test_acc 0.935 | past_task_test_acc 0.935                                                                                                                
INFO     epoch: 31, step 155 loss: 0.952 | val_loss 0.295 | train_acc 0.977 | val_acc 0.918 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                
INFO     epoch: 36, step 180 loss: 0.468 | val_loss 0.305 | train_acc 0.977 | val_acc 0.922 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                
INFO     epoch: 41, step 205 loss: 0.618 | val_loss 0.307 | train_acc 0.977 | val_acc 0.918 | test_acc 0.943 | past_task_test_acc 0.943                                                                                                                
INFO     epoch: 46, step 230 loss: 0.749 | val_loss 0.313 | train_acc 0.977 | val_acc 0.918 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                
INFO     epoch: 51, step 255 loss: 0.538 | val_loss 0.311 | train_acc 0.980 | val_acc 0.926 | test_acc 0.944 | past_task_test_acc 0.944                                                                                                                
INFO     epoch: 56, step 280 loss: 1.824 | val_loss 0.310 | train_acc 0.980 | val_acc 0.930 | test_acc 0.944 | past_task_test_acc 0.944                                                                                                                
INFO     epoch: 61, step 305 loss: 0.579 | val_loss 0.331 | train_acc 0.977 | val_acc 0.926 | test_acc 0.942 | past_task_test_acc 0.942                                                                                                                
INFO     epoch: 66, step 330 loss: 1.370 | val_loss 0.341 | train_acc 0.984 | val_acc 0.926 | test_acc 0.944 | past_task_test_acc 0.944                                                                                                                
INFO     epoch: 71, step 355 loss: 0.108 | val_loss 0.375 | train_acc 0.980 | val_acc 0.918 | test_acc 0.937 | past_task_test_acc 0.937                                                                                                                
INFO     epoch: 76, step 380 loss: 0.428 | val_loss 0.375 | train_acc 0.980 | val_acc 0.918 | test_acc 0.936 | past_task_test_acc 0.936                                                                                                                
INFO     epoch: 81, step 405 loss: 0.459 | val_loss 0.419 | train_acc 0.977 | val_acc 0.910 | test_acc 0.929 | past_task_test_acc 0.929                                                                                                                
INFO     epoch: 86, step 430 loss: 1.207 | val_loss 0.367 | train_acc 0.984 | val_acc 0.930 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                
INFO     epoch: 91, step 455 loss: 0.198 | val_loss 0.385 | train_acc 0.988 | val_acc 0.926 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                
INFO     epoch: 96, step 480 loss: 0.256 | val_loss 0.384 | train_acc 0.988 | val_acc 0.918 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                
INFO     epoch: 101, step 505 loss: 0.229 | val_loss 0.392 | train_acc 0.988 | val_acc 0.918 | test_acc 0.942 | past_task_test_acc 0.942                                                                                                               
INFO     epoch: 106, step 530 loss: 0.110 | val_loss 0.398 | train_acc 0.988 | val_acc 0.930 | test_acc 0.943 | past_task_test_acc 0.943                                                                                                               
INFO     Early stopping at epoch 106 with best val loss 0.070                                                                                                                                                                                          
CRITICAL Class 4 before 0.93359375 after 0.9296875 len 64 contribution: -0.0041841004184100415                                                                                                                                                         
DEBUG    No. of duplicates: 0                                                                                                                                                                                                                          
CRITICAL Buffer size: 320 cls distribution (tensor([0, 3, 4, 9]), tensor([ 64,  64,  64, 128]))                                                                                                                                                        
INFO     epoch: 1, step 5 loss: 2.541 | val_loss 0.228 | train_acc 0.973 | val_acc 0.922 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                   
INFO     epoch: 6, step 30 loss: 7.481 | val_loss 0.232 | train_acc 0.969 | val_acc 0.930 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                  
INFO     epoch: 11, step 55 loss: 1.560 | val_loss 0.264 | train_acc 0.973 | val_acc 0.918 | test_acc 0.929 | past_task_test_acc 0.929                                                                                                                 
INFO     epoch: 16, step 80 loss: 0.517 | val_loss 0.255 | train_acc 0.973 | val_acc 0.930 | test_acc 0.943 | past_task_test_acc 0.943                                                                                                                 
INFO     epoch: 21, step 105 loss: 0.313 | val_loss 0.292 | train_acc 0.977 | val_acc 0.922 | test_acc 0.933 | past_task_test_acc 0.933                                                                                                                
INFO     epoch: 26, step 130 loss: 0.194 | val_loss 0.279 | train_acc 0.977 | val_acc 0.930 | test_acc 0.941 | past_task_test_acc 0.941                                                                                                                
INFO     epoch: 31, step 155 loss: 0.405 | val_loss 0.300 | train_acc 0.980 | val_acc 0.930 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                
INFO     epoch: 36, step 180 loss: 0.569 | val_loss 0.323 | train_acc 0.984 | val_acc 0.930 | test_acc 0.937 | past_task_test_acc 0.937                                                                                                                
INFO     epoch: 41, step 205 loss: 0.848 | val_loss 0.340 | train_acc 0.984 | val_acc 0.926 | test_acc 0.936 | past_task_test_acc 0.936                                                                                                                
INFO     epoch: 46, step 230 loss: 0.444 | val_loss 0.340 | train_acc 0.980 | val_acc 0.926 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                
INFO     epoch: 51, step 255 loss: 1.182 | val_loss 0.362 | train_acc 0.984 | val_acc 0.930 | test_acc 0.940 | past_task_test_acc 0.940                                                                                                                
INFO     epoch: 56, step 280 loss: 1.184 | val_loss 0.350 | train_acc 0.984 | val_acc 0.938 | test_acc 0.938 | past_task_test_acc 0.938                                                                                                                
INFO     epoch: 61, step 305 loss: 0.175 | val_loss 0.383 | train_acc 0.988 | val_acc 0.934 | test_acc 0.938 | past_task_test_acc 0.938                                                                                                                
INFO     epoch: 66, step 330 loss: 0.151 | val_loss 0.412 | train_acc 0.988 | val_acc 0.930 | test_acc 0.934 | past_task_test_acc 0.934                                                                                                                
INFO     epoch: 71, step 355 loss: 0.085 | val_loss 0.411 | train_acc 0.984 | val_acc 0.930 | test_acc 0.937 | past_task_test_acc 0.937                                                                                                                
INFO     epoch: 76, step 380 loss: 0.212 | val_loss 0.450 | train_acc 0.992 | val_acc 0.918 | test_acc 0.930 | past_task_test_acc 0.930                                                                                                                
INFO     epoch: 81, step 405 loss: 0.091 | val_loss 0.431 | train_acc 0.988 | val_acc 0.930 | test_acc 0.935 | past_task_test_acc 0.935                                                                                                                
INFO     epoch: 86, step 430 loss: 0.188 | val_loss 0.480 | train_acc 0.992 | val_acc 0.918 | test_acc 0.929 | past_task_test_acc 0.929                                                                                                                
INFO     epoch: 91, step 455 loss: 0.036 | val_loss 0.458 | train_acc 0.988 | val_acc 0.930 | test_acc 0.934 | past_task_test_acc 0.934                                                                                                                
INFO     epoch: 96, step 480 loss: 0.222 | val_loss 0.504 | train_acc 0.988 | val_acc 0.918 | test_acc 0.929 | past_task_test_acc 0.929                                                                                                                
INFO     epoch: 101, step 505 loss: 0.009 | val_loss 0.540 | train_acc 0.984 | val_acc 0.910 | test_acc 0.925 | past_task_test_acc 0.925                                                                                                               
INFO     epoch: 106, step 530 loss: 0.102 | val_loss 0.502 | train_acc 0.988 | val_acc 0.930 | test_acc 0.931 | past_task_test_acc 0.931                                                                                                               
INFO     Early stopping at epoch 106 with best val loss 0.062                                                                                                                                                                                          
CRITICAL Class 9 before 0.93359375 after 0.9375 len 64 contribution: 0.0041841004184100415                                                                                                                                                             
DEBUG    No. of duplicates: 0                                                                                                                                                                                                                          
DEBUG    No. of duplicates: 0                                                                                                                                                                                                                          
Time taken: 64.3350465297699

__Testing__

Before the validation looks as expected. But somehow, validation stuff is all fucked uppp....


# TODO
tune the performance-based to beat the baseline with no-routing!


TODO:
debug this SHITTT.....

I DONT FUCKING UNDERSTAND WHY EVEN WITH VAL_SIZE VERY LARGE, WE STILL GET GARBAGE RESULT COMPARED TO

WE NEED TO VISUALIZE THE MISCLASSIFIED IMAGES AND ALSO THE IMAGES FROM THE SENDER!!

Maybe has to do Monte Carlo or something...





TODO:
keep the validation set. Might be have to up the train_size. Should do visualization on the entire
dataset (UMAP/TSNE), and then map the train/val/test of receiver and sender to that space to see if we're really seeing the desired effect! All of these can be tested on a fucking notebook. Might be try different MNIST architecture or even KNN on the embedding instead of NN.



Performance-based stuff is not looking too hot. Might be active learning would be much better!



https://rstudio-pubs-static.s3.amazonaws.com/354819_f31b02ac8fe84845b93e976a1271f463.html



Should we do up-weighting on the loss for the new data points???




https://machinelearningmastery.com/blending-ensemble-machine-learning-with-python/#:~:text=Blending%20is%20an%20ensemble%20machine%20learning%20technique%20that%20uses%20a,known%20as%20stacking%2C%20broadly%20conceived.


The small size of the validation is really hurting us...



# TODO:
https://www.kaggle.com/code/rafjaa/dealing-with-very-small-datasets/notebook
Use some sort of ensemble model to improve the accuracy / uncertainty quantification stuff...



Maybe get rid of early stop shit but training on fewer epochs!

Train a bunch and plot the results.
For image search, I'm concerned about the what distance metric to use...



# TODO
Bug in the sep stuff??


UMAP outlier detection is garbo or there's a bug somewhere??


What we want is some partial clustering that groups all the in-class together and can detect outliers (other classes!)



# README

https://pytorch-ood.readthedocs.io/en/latest/data.html
Bunch of data that we want like char24k


## TODO:
- implement autoencoder for the other baseline
- implement reservoir sampling with replay buffer. [x]



## TODO
Need to improve the outlier detection stuff...
Maybe ensemble a bunch of these detectors!


Use Meghna stuff...



Discrimator vs representation learning
Decouple represnetation & classifier

https://arxiv.org/pdf/2107.12753.pdf

https://arxiv.org/pdf/1910.09217.pdf


https://arxiv.org/abs/2009.08319



# TODO
use a separate data sharing buffer to reduce interference with known good class!



http://seba1511.net/tutorials/intermediate/dist_tuto.html
torch parallel


Actually, the lifelong ER might be implemented a bit wrong!

Might be a problem of not training enough??


Tricks to improve receiver_first:

- changed mean to upper percentile upper bound to improve the outlier detection! [DONE]
- Try other query strategy instead of wrongly_predicted to something else. [DONE],
will use wrongly + confidence
- Run other random seeds! Reduce the variance around the curve!
- We know that receiver ground truth is really good, which means that we need to further IMPROVE the outlier. (Probably not?) detection (using fancier open set recognition on MNIST!) [NOPE]
LAST RESORT...
- increase n_epochs from 200 to 500 to improve the image search quality.
- increase n_epochs from 20 to 50 to ensure convergence?

Thresholding tempering: https://github.com/KulikDM/pythresh




Mix 3 applications.

MNIST, FashionMNIST, KoreanMNIST.
Try to get this interleaved. Experiments.



Share linear combination of models and stuff...



5 components, have 

same init of components across agents.


SMOTE rebalance to work on fashionMNIST




At num_queries = 25
Class Multinomial resampler

Accuracy at task ll_time = 1
buffer train 10 epochs: 87%

buffer train 20 epochs (same as task) train more: 90%


SMOTE on raw features (flatten)
0.943945 (we're fucking back babyyyy!!)

SMOTE on encoded features
pretty bad lol... drop down to around 87%



NEXT:
Should increase num_queries back to 50 to further separate the curve between the baseline and ours! 

NOTE: might have to use some special loss from the deepSMOTE and/or VAE to make sure that the synthetic samples are properly reconstructed
(because vanilla autoencoder cannot ensure that!)



As noted by
Arjovsky et al. [13], many generative deep learning models
effectively incorporate a penalty, or noise, term in their loss
function, to impart diversity into the model distribution. For
example, both VAEs and WAEs include penalty terms in their
loss functions. We use permutation, instead of SMOTE, during
training because it is more memory and computationally
efficient
