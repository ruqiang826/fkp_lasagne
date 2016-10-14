# fkp_lasagne
facial keypoint detection from kaggle. code from http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

Model 1
==============

the first model is a simple model. 
>output:  
>    396       0.00225       0.00318      0.70744  0.05s    
>    397       0.00220       0.00329      0.66965  0.05s   
>    398       0.00231       0.00363      0.63604  0.05s   
>    399       0.00240       0.00361      0.66446  0.05s   
>    400       0.00233       0.00337      0.69174  0.05s   
> ...  
>    995       0.00139       0.00272      0.51203  0.05s  
>    996       0.00140       0.00271      0.51626  0.05s  
>    997       0.00137       0.00271      0.50514  0.05s  
>    998       0.00134       0.00276      0.48517  0.05s  
>    999       0.00135       0.00285      0.47278  0.05s  
>   1000       0.00139       0.00292      0.47477  0.05s  

if you see the picture of loss, you will find the model overfitting, because the train loss get better, but the valid loss do not change.

Model 2
==============
the second model use convolutional neural nets.It is much more computationally costly. In my GTX960 GPU, it runs for 24 mins.
> output:  
>    995       0.00112       0.00160      0.69759  1.35s  
>    996       0.00111       0.00160      0.69732  1.35s  
>    997       0.00111       0.00160      0.69688  1.35s  
>    998       0.00111       0.00160      0.69666  1.35s  
>    999       0.00111       0.00160      0.69617  1.35s  
>   1000       0.00111       0.00160      0.69575  1.35s  
> ...  
>   2993       0.00046       0.00143      0.32290  1.35s  
>   2994       0.00045       0.00142      0.31333  1.35s  
>   2995       0.00043       0.00143      0.29900  1.35s  
>   2996       0.00042       0.00145      0.28872  1.35s  
>   2997       0.00043       0.00149      0.28744  1.35s  
>   2998       0.00045       0.00151      0.29602  1.35s  
>   2999       0.00045       0.00150      0.30219  1.35s  
>   3000       0.00044       0.00148      0.30018  1.35s  

in model2, train loss and valid loss are both better than model1, after about 800 epochs, valid loss of model2 tend to horizontal.
![model1 2](https://cloud.githubusercontent.com/assets/22812703/19378940/1d821932-9222-11e6-9c23-b77159318032.png)


Model 3
==============
the third model is a example of Data augmentation. In the blog, Daniel simply flipping the images horizontically to get better performance. Still, it runs for 20+ mins for 1000 epochs, 1hour for 3000 epochs. 
> output:  
>     996       0.00136       0.00159      0.85147  1.37s  
>     997       0.00135       0.00159      0.84944  1.37s  
>     998       0.00136       0.00159      0.85629  1.37s  
>     999       0.00135       0.00159      0.85087  1.37s  
>    1000       0.00135       0.00159      0.84736  1.37s  
the advantage of model 3 is that it can get better valid loss after 3000 epochs,but at 1000 epochs, it is a little worse than model2, namely model3 have more potential, but computationally costly. How to fix it?
![model2 3](https://cloud.githubusercontent.com/assets/22812703/19378988/6502e6ce-9222-11e6-8b14-30a574a9de57.png)

Model 4 and 5
==============
model 4 and 5 use two methods to speed up model2 and 3. one is changing the iteration steps, namely to take big steps to learn quickly, lighter steps while we get to the optimum. the other is  increasing the optimization method's momentum parameter during training. It's the two parameter "update_learning_rate" and "update_momentum" in the code.

model4 is the faster version of model2, with 1000 ecpochs, it get better performance than model2 at epochs 3000
> output of model4:  
>     993       0.00046       0.00135      0.34096  1.35s  
>     994       0.00046       0.00135      0.34065  1.35s  
>     995       0.00046       0.00135      0.34033  1.35s  
>     996       0.00046       0.00135      0.34002  1.35s  
>     997       0.00046       0.00135      0.33970  1.35s  
>     998       0.00046       0.00135      0.33938  1.35s  
>     999       0.00046       0.00135      0.33906  1.35s  
>    1000       0.00046       0.00135      0.33874  1.35s  

so does model5 vs model3. valid loss of "0.00118" is the best outcome until now.
>  output of model5:  
>     993       0.00071       0.00118      0.60333  1.37s  
>     994       0.00072       0.00118      0.60583  1.37s  
>     995       0.00072       0.00118      0.60804  1.36s  
>     996       0.00072       0.00118      0.60747  1.37s  
>     997       0.00071       0.00118      0.60454  1.37s  
>     998       0.00071       0.00118      0.60314  1.37s  
>     999       0.00072       0.00118      0.60615  1.37s  
>    1000       0.00072       0.00118      0.60537  1.37s  



