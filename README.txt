## Experimental environment

+ python >= 3.8
+ pytorch >= 1.10.1
+ numpy
+ pandas
+ tqdm
+ sklearn
+ multiprocessing
+ timm
+sys

## This paper is experimentally for 
1. SLaK/main.py -- Extract feature of  images. For example, given  the prtrained model and extract feature of  images.
+ eval: bool, which perform evaluation only to extract feature
+ resume: path to the prtrained model 
+ data_set:  dataset name
+ data_path: path to the dataset 
+ image_in_channels: int, the number of channels to enter the image       
+ time_step: an integer indicating a move time step
+ num_stack: int, Convlstm number of stacks
+ batch_size: int
+ lr: float, such as 4e-3
+ epochs: int, epoch to train

2. BCV_valid.py-- After loading the extracted features, integrate them using the proposed BCV method
+ m: int, which sets the number of divisions
3. CV_valid. py -- After loading the extracted features, integrate them using the proposed CV method
+ k_fold: int, which sets the number of divisions


The code address of this paper is http://github.com/MICVRGroup/FEN-basedCV.
The code address of SLaK is https://github.com/VITA-Group/SLaK.
The code address of mobilenet is https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py
The code address of efficientnet_v2 is https://github.com/google/automl/tree/master
The code address of SuperWeights is https://github.com/piotr-teterwak/SuperWeights


The code will be provided free of charge to meteorology-related researchers in order 
to promote research. Without prior approval from the providers, thiscode will not be used for commercial use.  

In all documents and papers that report experimental results based on the paper, 
a citation of this paper should be added into the references or acknowledged in the acknowledgement.

 


 
