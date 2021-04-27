# FCN-pytorch
FCN-pytorch implement<br>  
for train you show download the voc11 dataset  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
 
and then you zip it in the datasets roots<br>  
best important thing in here is the learning rate you must set 1e-6,decay by 10 for [40,50] stoneline.  
all the optim you can set in config.py .<br>
python main.py train for train your dataset .<br>
python main.py test for test the image ,and save output in resuls files.<br>
if you have question.<br>  
