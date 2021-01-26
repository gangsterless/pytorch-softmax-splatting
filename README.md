# pytorch-soft-splatting
personal reimplementation of softsplatting


# pytorch-soft-splatting
First apologize for my poor English

This is a personal reimplementation of softsplatting [1] using PyTorch. The code is complete comparing with official code. That's to say it is trainable and I will provide a pretrained model later.

Please NOTE that I am a beginner in deep learning especially in frame interpolation. And the most important thing is that I only have a 1060 GPU.(so poor right?) so my model is trained based on a subset of Vimeo90k including about 1600 triples. I am unable to train the whole dataset currently, so more test and criteria is not available now. And I am not so sure of some training details. 
 
 But there are still some good news. I think it is easy for you to train and run your own model. Also, the code is purely python your can feel free of issues of cuda version,gcc version and platform(windows and linux both run well). Despite of shortage of test, my reimplementation looks at least plausible.
 # examples
 left is interpolation result and right is ground truth.<br/>some very good res.

 <p align="center"><img src="./example_img/test7.jpg" alt="Comparison"></p>
 <p align="center"><img src="./example_img/test4.jpg" alt="Comparison"></p>

<br/>some bad res:
<p align="center"><img src="./example_img/test10.jpg" alt="Comparison"></p>
you can clearly see some overlap and blur in the left.

## environment
 I run in python36 cupy 10.1 in fact I think it can also run well in any version
 for python>=3.6.2

## usage
### to simply test my code you should
1 download my pretrained model from
[baiduyun]()<br>

2 run :
```
python run_a_pair.py
```
### to train my code you should
1 download Vimeo <br>
2 download  pretrained model of pwc and move it to 'pwc\weights\network-default.pytorch'<br>
[baiduyun]() <br>if you want to train from scratch you can choose not to download this and comment the **torch.load** but you may get very bad result. <br>
3 change the variable **vimo_data_dir** in train_and_test.py  to 'path_to_your_dataset/vimeo_triplet'<br>

4  I didn't write args parser  for your convenience to revise my code. Change the settings as what you like in train_or_test.It 's very easy to revise. or just run as:

```
python train_and_test.py
```
IF YOU HAVE BETTER DEVICE AND YOU CAN TRAIN WHOLE DATASET PLEASE TELL ME THANK YOU VERY MUCH!!!


## structure of code
The code mainly include 3 parts <br>
  1  gridnet[3]:  it's a U-net like structure network which is used to compile the img1,img2 and their warped results. for more info please see  the code or the papar.Note you can train this part without any other parts. it may serve as a baseline  <br> 
  2 pwc[2]: It's a network that utilizes to extract flow from images. for more info please see  the code or the papar.   This is also trainable because I wrote the train code.But for my device limitation, I cannot fine-tune the pretrained model pwc\weights\network-default.pytorch. If you can fine-tune this on flythings or flychairs I think you can get better result(this is advice from author)<br>
  3 softsplatting[1]: the 'soft operation'.<br>
  4 main_net: put all things metioned above together.




## references
```
[1]  @inproceedings{Niklaus_CVPR_2020,
         author = {Simon Niklaus and Feng Liu},
         title = {Softmax Splatting for Video Frame Interpolation},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2020}
     }
```
[2] [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc) <br>
[3] [grid-net](https://github.com/daigo0927/GridNet)

Many codes are from Niklaus.<br>
Many Many Many thanks for him.


