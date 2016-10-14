# fkp_lasagne
facial keypoint detection from kaggle. code from http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

Model 1
==============

the first model is a simple model. 
>output:  >    396       0.00225       0.00318      0.70744  0.05s    >    397       0.00220       0.00329      0.66965  0.05s   
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
> ...  
>    2993       0.00073       0.00128      0.56746  1.40s  
>    2994       0.00072       0.00128      0.56492  1.38s  
>    2995       0.00072       0.00129      0.56124  1.46s  
>    2996       0.00073       0.00128      0.56692  1.41s  
>    2997       0.00072       0.00128      0.56616  1.44s  
>    2998       0.00072       0.00128      0.56334  1.41s  
>    2999       0.00072       0.00128      0.56134  1.37s  
>    3000       0.00072       0.00128      0.56403  1.37s  

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

the comparision of model 4 and 5:
![model45](https://cloud.githubusercontent.com/assets/22812703/19381345/c180aade-922d-11e6-9a0d-380d70505ec5.png)


Model 6 and 7 using dropout
==============
Although we get a good result, there is still a problem of overfitting(train loss is significantly better than valid loss).`Dropout is a popular regularization technique for neural network, which is implemented in model6 and improved in model7.  
We go directly to model7.

> loss of model 7:  
> Epoch  |  Train loss  |  Valid loss  |  Train / Val  
> --------|--------------|--------------|---------------  
>     50  |    0.004756  |    0.007043  |     0.675330   
>    100  |    0.004440  |    0.005321  |     0.834432   
>    250  |    0.003974  |    0.003928  |     1.011598   
>    500  |    0.002574  |    0.002347  |     1.096366   
>   1000  |    0.001861  |    0.001613  |     1.153796   
>   1500  |    0.001558  |    0.001372  |     1.135849   
>   2000  |    0.001409  |    0.001230  |     1.144821   
>   2500  |    0.001295  |    0.001146  |     1.130188   
>   3000  |    0.001195  |    0.001087  |     1.099271   

the valid loss of model 7 in epoch 1000 is worse than model 5. but better than model 5 in epoch 3000. Another improvement is even at epoch 3000, model 7 do not overfit, which means we can do more iteration using model7.

we set the max_epochs to 10000, the output is:

>  Epoch  |  Train loss  |  Valid loss  |  Train / Val  
> --------|--------------|--------------|---------------  
>     50  |    0.004756  |    0.007027  |     0.676810  
>    100  |    0.004439  |    0.005321  |     0.834323  
>    500  |    0.002576  |    0.002346  |     1.097795  
>   1000  |    0.001863  |    0.001614  |     1.154038  
>   2000  |    0.001406  |    0.001233  |     1.140188  
>   3000  |    0.001184  |    0.001074  |     1.102168  
>   4000  |    0.001068  |    0.000983  |     1.086193  
>   5000  |    0.000981  |    0.000920  |     1.066288  
>   6000  |    0.000904  |    0.000884  |     1.021837  
>   7000  |    0.000851  |    0.000849  |     1.002314  
>   8000  |    0.000810  |    0.000821  |     0.985769  
>   9000  |    0.000769  |    0.000803  |     0.957842  
>  10000  |    0.000760  |    0.000787  |     0.966583  

