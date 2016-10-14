# fkp_lasagne
facial keypoint detection from kaggle. code from http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

the first model is a simple model. 
>output:
>    396       0.00225       0.00318      0.70744  0.05s
>    397       0.00220       0.00329      0.66965  0.05s
>    398       0.00231       0.00363      0.63604  0.05s
>    399       0.00240       0.00361      0.66446  0.05s
>    400       0.00233       0.00337      0.69174  0.05s
if you see the picture of loss, you will find the model overfitting, because the train loss get better, but the valid loss do not change.


the second model use convolutional neural nets.It is much more computationally costly. In my GTX960 GPU, it runs for 24 mins.
the performance:
    996       0.00114       0.00164      0.69813  1.35s
    997       0.00114       0.00163      0.69782  1.35s
    998       0.00114       0.00163      0.69732  1.35s
    999       0.00114       0.00163      0.69692  1.36s
   1000       0.00114       0.00163      0.69638  1.36s

for model1 and model2,
![model1 2](https://cloud.githubusercontent.com/assets/22812703/19378988/6502e6ce-9222-11e6-8b14-30a574a9de57.png)

the third model is a example of Data augmentation, flipping the images horizontically to get better performance. Still, it runs for 20+ mins for 1000 epochs.  model 3 get better valid loss because of data augmentation.
the performance:
    996       0.00136       0.00159      0.85147  1.37s
    997       0.00135       0.00159      0.84944  1.37s
    998       0.00136       0.00159      0.85629  1.37s
    999       0.00135       0.00159      0.85087  1.37s
   1000       0.00135       0.00159      0.84736  1.37s

  

model 4 use two methods to make the train faster. one is changing the iteration steps, namely to take big steps to learn quickly, lighter steps while we get to the optimum. the other is  increasing the optimization method's momentum parameter during training. It's the two parameter "update_learning_rate" and "update_momentum" in the code.


