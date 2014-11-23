---
title: "Practical Machine Learning - Course Project"
output: html_document
---


## Executive summary
The objective of this project is to built a predictive model which, based on data from accelerometers on the belt, forearm  and dumbbell of participant, will be able to evaluate if a weight lifting movement is well executed. 

The document that follows, gives all the steps that were taken as well as explain the different choices that were made to build the this predictive model


## Initial steps

First , we install the required packages and load the required libraries.


```r
##Install.packages("caret")
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## 
## Attaching package: 'caret'
## 
## The following object is masked _by_ '.GlobalEnv':
## 
##     best
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

We read the provided data which will be used to create the training and testing datasets.


```r
ds<-read.csv("pml-training.csv")
```

## Predictor selection

The provided dataset ("pml-training.csv") has one outcome, 159 predictors and over 19000 lines for 4 separate sensors. 

In this section, in an effort to reduce the amount of data to process, we are going to try to look at the different predictors to evaluate if they have an added value to the predictive model we are trying to build

### Time related predictors

We have chosen to ignore  all time related predictors. The model that we are trying to built has no notion of time and we don't feel they have an added value in this particular context.

### Belt sensor predictors

The belt sensor has 38 distinct predictors.


```r
belt_subset <- ds[,c(8:45)]
summary(belt_subset)
```

```
##    roll_belt       pitch_belt        yaw_belt      total_accel_belt
##  Min.   :-28.9   Min.   :-55.80   Min.   :-180.0   Min.   : 0.0    
##  1st Qu.:  1.1   1st Qu.:  1.76   1st Qu.: -88.3   1st Qu.: 3.0    
##  Median :113.0   Median :  5.28   Median : -13.0   Median :17.0    
##  Mean   : 64.4   Mean   :  0.31   Mean   : -11.2   Mean   :11.3    
##  3rd Qu.:123.0   3rd Qu.: 14.90   3rd Qu.:  12.9   3rd Qu.:18.0    
##  Max.   :162.0   Max.   : 60.30   Max.   : 179.0   Max.   :29.0    
##                                                                    
##  kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
##           :19216             :19216            :19216    
##  #DIV/0!  :   10    #DIV/0!  :   32     #DIV/0!:  406    
##  -1.908453:    2    47       :    4                      
##  -0.01685 :    1    -0.15095 :    3                      
##  -0.021024:    1    -0.684748:    3                      
##  -0.025513:    1    -1.750749:    3                      
##  (Other)  :  391    (Other)  :  361                      
##  skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt  
##           :19216             :19216             :19216     Min.   :-94    
##  #DIV/0!  :    9    #DIV/0!  :   32      #DIV/0!:  406     1st Qu.:-88    
##  0        :    4    0        :    4                        Median : -5    
##  0.422463 :    2    -2.156553:    3                        Mean   : -7    
##  -0.003095:    1    -3.072669:    3                        3rd Qu.: 18    
##  -0.010002:    1    -6.324555:    3                        Max.   :180    
##  (Other)  :  389    (Other)  :  361                        NA's   :19216  
##  max_picth_belt   max_yaw_belt   min_roll_belt   min_pitch_belt 
##  Min.   : 3             :19216   Min.   :-180    Min.   : 0     
##  1st Qu.: 5      -1.1   :   30   1st Qu.: -88    1st Qu.: 3     
##  Median :18      -1.4   :   29   Median :  -8    Median :16     
##  Mean   :13      -1.2   :   26   Mean   : -10    Mean   :11     
##  3rd Qu.:19      -0.9   :   24   3rd Qu.:   9    3rd Qu.:17     
##  Max.   :30      -1.3   :   22   Max.   : 173    Max.   :23     
##  NA's   :19216   (Other):  275   NA's   :19216   NA's   :19216  
##   min_yaw_belt   amplitude_roll_belt amplitude_pitch_belt
##         :19216   Min.   :  0         Min.   : 0          
##  -1.1   :   30   1st Qu.:  0         1st Qu.: 1          
##  -1.4   :   29   Median :  1         Median : 1          
##  -1.2   :   26   Mean   :  4         Mean   : 2          
##  -0.9   :   24   3rd Qu.:  2         3rd Qu.: 2          
##  -1.3   :   22   Max.   :360         Max.   :12          
##  (Other):  275   NA's   :19216       NA's   :19216       
##  amplitude_yaw_belt var_total_accel_belt avg_roll_belt   stddev_roll_belt
##         :19216      Min.   : 0           Min.   :-27     Min.   : 0      
##  #DIV/0!:   10      1st Qu.: 0           1st Qu.:  1     1st Qu.: 0      
##  0      :  396      Median : 0           Median :116     Median : 0      
##                     Mean   : 1           Mean   : 68     Mean   : 1      
##                     3rd Qu.: 0           3rd Qu.:123     3rd Qu.: 1      
##                     Max.   :16           Max.   :157     Max.   :14      
##                     NA's   :19216        NA's   :19216   NA's   :19216   
##  var_roll_belt   avg_pitch_belt  stddev_pitch_belt var_pitch_belt 
##  Min.   :  0     Min.   :-51     Min.   :0         Min.   : 0     
##  1st Qu.:  0     1st Qu.:  2     1st Qu.:0         1st Qu.: 0     
##  Median :  0     Median :  5     Median :0         Median : 0     
##  Mean   :  8     Mean   :  1     Mean   :1         Mean   : 1     
##  3rd Qu.:  0     3rd Qu.: 16     3rd Qu.:1         3rd Qu.: 0     
##  Max.   :201     Max.   : 60     Max.   :4         Max.   :16     
##  NA's   :19216   NA's   :19216   NA's   :19216     NA's   :19216  
##   avg_yaw_belt   stddev_yaw_belt  var_yaw_belt    gyros_belt_x    
##  Min.   :-138    Min.   :  0     Min.   :    0   Min.   :-1.0400  
##  1st Qu.: -88    1st Qu.:  0     1st Qu.:    0   1st Qu.:-0.0300  
##  Median :  -7    Median :  0     Median :    0   Median : 0.0300  
##  Mean   :  -9    Mean   :  1     Mean   :  107   Mean   :-0.0056  
##  3rd Qu.:  14    3rd Qu.:  1     3rd Qu.:    0   3rd Qu.: 0.1100  
##  Max.   : 174    Max.   :177     Max.   :31183   Max.   : 2.2200  
##  NA's   :19216   NA's   :19216   NA's   :19216                    
##   gyros_belt_y      gyros_belt_z     accel_belt_x      accel_belt_y  
##  Min.   :-0.6400   Min.   :-1.460   Min.   :-120.00   Min.   :-69.0  
##  1st Qu.: 0.0000   1st Qu.:-0.200   1st Qu.: -21.00   1st Qu.:  3.0  
##  Median : 0.0200   Median :-0.100   Median : -15.00   Median : 35.0  
##  Mean   : 0.0396   Mean   :-0.131   Mean   :  -5.59   Mean   : 30.1  
##  3rd Qu.: 0.1100   3rd Qu.:-0.020   3rd Qu.:  -5.00   3rd Qu.: 61.0  
##  Max.   : 0.6400   Max.   : 1.620   Max.   :  85.00   Max.   :164.0  
##                                                                      
##   accel_belt_z    magnet_belt_x   magnet_belt_y magnet_belt_z 
##  Min.   :-275.0   Min.   :-52.0   Min.   :354   Min.   :-623  
##  1st Qu.:-162.0   1st Qu.:  9.0   1st Qu.:581   1st Qu.:-375  
##  Median :-152.0   Median : 35.0   Median :601   Median :-320  
##  Mean   : -72.6   Mean   : 55.6   Mean   :594   Mean   :-346  
##  3rd Qu.:  27.0   3rd Qu.: 59.0   3rd Qu.:610   3rd Qu.:-306  
##  Max.   : 105.0   Max.   :485.0   Max.   :673   Max.   : 293  
## 
```

We can see that many of the predictors are very sparsely populated. In fact 25 of the 38 predictor are either empty or populated by  the value "N/A" nearly 98% of the time. These sparsely populated predictors seem to have a value only when they are associated to a record of type a "New window". 

Due to their sparsity and due to the fact that the type of record  "New window" is not present in the test dataset we have chosen to eliminate these predictors.

We then keep only 13 of the 38 predictors associated to the belt sensor. The list of  kept predictors is given below.


```r
valid_belt_predictors <- c(8:11,37:45)
names(ds[,valid_belt_predictors])
```

```
##  [1] "roll_belt"        "pitch_belt"       "yaw_belt"        
##  [4] "total_accel_belt" "gyros_belt_x"     "gyros_belt_y"    
##  [7] "gyros_belt_z"     "accel_belt_x"     "accel_belt_y"    
## [10] "accel_belt_z"     "magnet_belt_x"    "magnet_belt_y"   
## [13] "magnet_belt_z"
```

### Arm sensors predictors

The arm sensor has also 38 predictors. The pattern of the data associated to theses sensors is similar to what was described for the belt sensors. We have then, for the same reasons as given for the belt sensor, decided to reject the predictors that were only populated for the records of type "New window". 


```r
arm_subset <- ds[,c(46:83)]
summary(arm_subset)
```

```
##     roll_arm        pitch_arm         yaw_arm        total_accel_arm
##  Min.   :-180.0   Min.   :-88.80   Min.   :-180.00   Min.   : 1.0   
##  1st Qu.: -31.8   1st Qu.:-25.90   1st Qu.: -43.10   1st Qu.:17.0   
##  Median :   0.0   Median :  0.00   Median :   0.00   Median :27.0   
##  Mean   :  17.8   Mean   : -4.61   Mean   :  -0.62   Mean   :25.5   
##  3rd Qu.:  77.3   3rd Qu.: 11.20   3rd Qu.:  45.88   3rd Qu.:33.0   
##  Max.   : 180.0   Max.   : 88.50   Max.   : 180.00   Max.   :66.0   
##                                                                     
##  var_accel_arm    avg_roll_arm   stddev_roll_arm  var_roll_arm  
##  Min.   :  0     Min.   :-167    Min.   :  0     Min.   :    0  
##  1st Qu.:  9     1st Qu.: -38    1st Qu.:  1     1st Qu.:    2  
##  Median : 41     Median :   0    Median :  6     Median :   33  
##  Mean   : 53     Mean   :  13    Mean   : 11     Mean   :  417  
##  3rd Qu.: 76     3rd Qu.:  76    3rd Qu.: 15     3rd Qu.:  223  
##  Max.   :332     Max.   : 163    Max.   :162     Max.   :26232  
##  NA's   :19216   NA's   :19216   NA's   :19216   NA's   :19216  
##  avg_pitch_arm   stddev_pitch_arm var_pitch_arm    avg_yaw_arm   
##  Min.   :-82     Min.   : 0       Min.   :   0    Min.   :-173   
##  1st Qu.:-23     1st Qu.: 2       1st Qu.:   3    1st Qu.: -29   
##  Median :  0     Median : 8       Median :  66    Median :   0   
##  Mean   : -5     Mean   :10       Mean   : 196    Mean   :   2   
##  3rd Qu.:  8     3rd Qu.:16       3rd Qu.: 267    3rd Qu.:  38   
##  Max.   : 76     Max.   :43       Max.   :1885    Max.   : 152   
##  NA's   :19216   NA's   :19216    NA's   :19216   NA's   :19216  
##  stddev_yaw_arm   var_yaw_arm     gyros_arm_x      gyros_arm_y    
##  Min.   :  0     Min.   :    0   Min.   :-6.370   Min.   :-3.440  
##  1st Qu.:  3     1st Qu.:    7   1st Qu.:-1.330   1st Qu.:-0.800  
##  Median : 17     Median :  278   Median : 0.080   Median :-0.240  
##  Mean   : 22     Mean   : 1056   Mean   : 0.043   Mean   :-0.257  
##  3rd Qu.: 36     3rd Qu.: 1295   3rd Qu.: 1.570   3rd Qu.: 0.140  
##  Max.   :177     Max.   :31345   Max.   : 4.870   Max.   : 2.840  
##  NA's   :19216   NA's   :19216                                    
##   gyros_arm_z     accel_arm_x      accel_arm_y      accel_arm_z    
##  Min.   :-2.33   Min.   :-404.0   Min.   :-318.0   Min.   :-636.0  
##  1st Qu.:-0.07   1st Qu.:-242.0   1st Qu.: -54.0   1st Qu.:-143.0  
##  Median : 0.23   Median : -44.0   Median :  14.0   Median : -47.0  
##  Mean   : 0.27   Mean   : -60.2   Mean   :  32.6   Mean   : -71.2  
##  3rd Qu.: 0.72   3rd Qu.:  84.0   3rd Qu.: 139.0   3rd Qu.:  23.0  
##  Max.   : 3.02   Max.   : 437.0   Max.   : 308.0   Max.   : 292.0  
##                                                                    
##   magnet_arm_x   magnet_arm_y   magnet_arm_z  kurtosis_roll_arm
##  Min.   :-584   Min.   :-392   Min.   :-597           :19216   
##  1st Qu.:-300   1st Qu.:  -9   1st Qu.: 131   #DIV/0! :   78   
##  Median : 289   Median : 202   Median : 444   -0.02438:    1   
##  Mean   : 192   Mean   : 157   Mean   : 306   -0.0419 :    1   
##  3rd Qu.: 637   3rd Qu.: 323   3rd Qu.: 545   -0.05051:    1   
##  Max.   : 782   Max.   : 583   Max.   : 694   -0.05695:    1   
##                                               (Other) :  324   
##  kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm
##          :19216             :19216           :19216            :19216    
##  #DIV/0! :   80     #DIV/0! :   11   #DIV/0! :   77    #DIV/0! :   80    
##  -0.00484:    1     0.55844 :    2   -0.00051:    1    -0.00184:    1    
##  -0.01311:    1     0.65132 :    2   -0.00696:    1    -0.01185:    1    
##  -0.02967:    1     -0.01548:    1   -0.01884:    1    -0.01247:    1    
##  -0.07394:    1     -0.01749:    1   -0.03359:    1    -0.02063:    1    
##  (Other) :  322     (Other) :  389   (Other) :  325    (Other) :  322    
##  skewness_yaw_arm  max_roll_arm   max_picth_arm    max_yaw_arm   
##          :19216   Min.   :-73     Min.   :-173    Min.   : 4     
##  #DIV/0! :   11   1st Qu.:  0     1st Qu.:  -2    1st Qu.:29     
##  -1.62032:    2   Median :  5     Median :  23    Median :34     
##  0.55053 :    2   Mean   : 11     Mean   :  36    Mean   :35     
##  -0.00311:    1   3rd Qu.: 27     3rd Qu.:  96    3rd Qu.:41     
##  -0.00562:    1   Max.   : 86     Max.   : 180    Max.   :65     
##  (Other) :  389   NA's   :19216   NA's   :19216   NA's   :19216  
##   min_roll_arm   min_pitch_arm    min_yaw_arm    amplitude_roll_arm
##  Min.   :-89     Min.   :-180    Min.   : 1      Min.   :  0       
##  1st Qu.:-42     1st Qu.: -73    1st Qu.: 8      1st Qu.:  5       
##  Median :-22     Median : -34    Median :13      Median : 28       
##  Mean   :-21     Mean   : -34    Mean   :15      Mean   : 32       
##  3rd Qu.:  0     3rd Qu.:   0    3rd Qu.:19      3rd Qu.: 51       
##  Max.   : 66     Max.   : 152    Max.   :38      Max.   :120       
##  NA's   :19216   NA's   :19216   NA's   :19216   NA's   :19216     
##  amplitude_pitch_arm amplitude_yaw_arm
##  Min.   :  0         Min.   : 0       
##  1st Qu.: 10         1st Qu.:13       
##  Median : 55         Median :22       
##  Mean   : 70         Mean   :21       
##  3rd Qu.:115         3rd Qu.:29       
##  Max.   :360         Max.   :52       
##  NA's   :19216       NA's   :19216
```

Again, we were left with 13 sensors out of 38. The list of sensor that will be considered in the predictive model is given below.


```r
valid_arm_predictors <- c(46:49,60:68)
names(ds[,valid_arm_predictors])
```

```
##  [1] "roll_arm"        "pitch_arm"       "yaw_arm"        
##  [4] "total_accel_arm" "gyros_arm_x"     "gyros_arm_y"    
##  [7] "gyros_arm_z"     "accel_arm_x"     "accel_arm_y"    
## [10] "accel_arm_z"     "magnet_arm_x"    "magnet_arm_y"   
## [13] "magnet_arm_z"
```

### Dumbbell sensors predictors

The dumbbell sensor has also 38 predictors, and for the same reasons mentioned in the previous two cases we have also rejected the predictors who were populated only in records of type "New window".


```r
dumbbell_subset <- ds[,c(84:121)]
summary(dumbbell_subset)
```

```
##  roll_dumbbell    pitch_dumbbell    yaw_dumbbell    
##  Min.   :-153.7   Min.   :-149.6   Min.   :-150.87  
##  1st Qu.: -18.5   1st Qu.: -40.9   1st Qu.: -77.64  
##  Median :  48.2   Median : -21.0   Median :  -3.32  
##  Mean   :  23.8   Mean   : -10.8   Mean   :   1.67  
##  3rd Qu.:  67.6   3rd Qu.:  17.5   3rd Qu.:  79.64  
##  Max.   : 153.6   Max.   : 149.4   Max.   : 154.95  
##                                                     
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
##         :19216                 :19216                  :19216        
##  #DIV/0!:    5          -0.5464:    2           #DIV/0!:  406        
##  -0.2583:    2          -0.9334:    2                                
##  -0.3705:    2          -2.0833:    2                                
##  -0.5855:    2          -2.0851:    2                                
##  -2.0851:    2          -2.0889:    2                                
##  (Other):  393          (Other):  396                                
##  skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
##         :19216                 :19216                  :19216        
##  #DIV/0!:    4          -0.2328:    2           #DIV/0!:  406        
##  -0.9324:    2          -0.3521:    2                                
##  0.111  :    2          -0.7036:    2                                
##  1.0312 :    2          0.109  :    2                                
##  -0.0082:    1          1.0326 :    2                                
##  (Other):  395          (Other):  396                                
##  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
##  Min.   :-70       Min.   :-113              :19216    Min.   :-150     
##  1st Qu.:-27       1st Qu.: -67       -0.6   :   20    1st Qu.: -60     
##  Median : 15       Median :  40       0.2    :   19    Median : -44     
##  Mean   : 14       Mean   :  33       -0.8   :   18    Mean   : -41     
##  3rd Qu.: 51       3rd Qu.: 133       -0.3   :   16    3rd Qu.: -25     
##  Max.   :137       Max.   : 155       -0.2   :   15    Max.   :  73     
##  NA's   :19216     NA's   :19216      (Other):  318    NA's   :19216    
##  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
##  Min.   :-147              :19216    Min.   :  0            
##  1st Qu.: -92       -0.6   :   20    1st Qu.: 15            
##  Median : -66       0.2    :   19    Median : 35            
##  Mean   : -33       -0.8   :   18    Mean   : 55            
##  3rd Qu.:  21       -0.3   :   16    3rd Qu.: 81            
##  Max.   : 121       -0.2   :   15    Max.   :256            
##  NA's   :19216      (Other):  318    NA's   :19216          
##  amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
##  Min.   :  0                     :19216          Min.   : 0.0        
##  1st Qu.: 17              #DIV/0!:    5          1st Qu.: 4.0        
##  Median : 42              0      :  401          Median :10.0        
##  Mean   : 66                                     Mean   :13.7        
##  3rd Qu.:100                                     3rd Qu.:19.0        
##  Max.   :274                                     Max.   :58.0        
##  NA's   :19216                                                       
##  var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
##  Min.   :  0        Min.   :-129      Min.   :  0         
##  1st Qu.:  0        1st Qu.: -12      1st Qu.:  5         
##  Median :  1        Median :  48      Median : 12         
##  Mean   :  4        Mean   :  24      Mean   : 21         
##  3rd Qu.:  3        3rd Qu.:  64      3rd Qu.: 26         
##  Max.   :230        Max.   : 126      Max.   :124         
##  NA's   :19216      NA's   :19216     NA's   :19216       
##  var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
##  Min.   :    0     Min.   :-71        Min.   : 0           
##  1st Qu.:   22     1st Qu.:-42        1st Qu.: 3           
##  Median :  149     Median :-20        Median : 8           
##  Mean   : 1020     Mean   :-12        Mean   :13           
##  3rd Qu.:  695     3rd Qu.: 13        3rd Qu.:19           
##  Max.   :15321     Max.   : 94        Max.   :83           
##  NA's   :19216     NA's   :19216      NA's   :19216        
##  var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
##  Min.   :   0       Min.   :-118     Min.   :  0         Min.   :    0   
##  1st Qu.:  12       1st Qu.: -77     1st Qu.:  4         1st Qu.:   15   
##  Median :  65       Median :  -5     Median : 10         Median :  105   
##  Mean   : 350       Mean   :   0     Mean   : 17         Mean   :  590   
##  3rd Qu.: 370       3rd Qu.:  71     3rd Qu.: 25         3rd Qu.:  609   
##  Max.   :6836       Max.   : 135     Max.   :107         Max.   :11468   
##  NA's   :19216      NA's   :19216    NA's   :19216       NA's   :19216   
##  gyros_dumbbell_x  gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
##  Min.   :-204.00   Min.   :-2.10    Min.   : -2.4    Min.   :-419.0  
##  1st Qu.:  -0.03   1st Qu.:-0.14    1st Qu.: -0.3    1st Qu.: -50.0  
##  Median :   0.13   Median : 0.03    Median : -0.1    Median :  -8.0  
##  Mean   :   0.16   Mean   : 0.05    Mean   : -0.1    Mean   : -28.6  
##  3rd Qu.:   0.35   3rd Qu.: 0.21    3rd Qu.:  0.0    3rd Qu.:  11.0  
##  Max.   :   2.22   Max.   :52.00    Max.   :317.0    Max.   : 235.0  
##                                                                      
##  accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
##  Min.   :-189.0   Min.   :-334.0   Min.   :-643      Min.   :-3600    
##  1st Qu.:  -8.0   1st Qu.:-142.0   1st Qu.:-535      1st Qu.:  231    
##  Median :  41.5   Median :  -1.0   Median :-479      Median :  311    
##  Mean   :  52.6   Mean   : -38.3   Mean   :-328      Mean   :  221    
##  3rd Qu.: 111.0   3rd Qu.:  38.0   3rd Qu.:-304      3rd Qu.:  390    
##  Max.   : 315.0   Max.   : 318.0   Max.   : 592      Max.   :  633    
##                                                                       
##  magnet_dumbbell_z
##  Min.   :-262.0   
##  1st Qu.: -45.0   
##  Median :  13.0   
##  Mean   :  46.1   
##  3rd Qu.:  95.0   
##  Max.   : 452.0   
## 
```

Again, we were left with 13 sensors out of 38. The list of sensor that will be considered in the predictive model is given below.


```r
valid_dumbbell_predictors <- c(84:86, 102, 113:121)
names(ds[,valid_dumbbell_predictors])
```

```
##  [1] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
##  [4] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
##  [7] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [10] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [13] "magnet_dumbbell_z"
```

### Forearm sensors  predictors

Finally, the forearm sensor has also 38 predictors, and again for the reasons mentioned in the previous three cases we have also rejected the predictors who were populated only in records of type "New window".



```r
forearm_subset <- ds[,c(122:159)]
summary(forearm_subset)
```

```
##   roll_forearm     pitch_forearm     yaw_forearm     kurtosis_roll_forearm
##  Min.   :-180.00   Min.   :-72.50   Min.   :-180.0          :19216        
##  1st Qu.:  -0.74   1st Qu.:  0.00   1st Qu.: -68.6   #DIV/0!:   84        
##  Median :  21.70   Median :  9.24   Median :   0.0   -0.8079:    2        
##  Mean   :  33.83   Mean   : 10.71   Mean   :  19.2   -0.9169:    2        
##  3rd Qu.: 140.00   3rd Qu.: 28.40   3rd Qu.: 110.0   -0.0227:    1        
##  Max.   : 180.00   Max.   : 89.80   Max.   : 180.0   -0.0359:    1        
##                                                      (Other):  316        
##  kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
##         :19216                 :19216               :19216        
##  #DIV/0!:   85          #DIV/0!:  406        #DIV/0!:   83        
##  -0.0073:    1                               -0.1912:    2        
##  -0.0442:    1                               -0.4126:    2        
##  -0.0489:    1                               -0.0004:    1        
##  -0.0523:    1                               -0.0013:    1        
##  (Other):  317                               (Other):  317        
##  skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
##         :19216                 :19216        Min.   :-67     
##  #DIV/0!:   85          #DIV/0!:  406        1st Qu.:  0     
##  0      :    4                               Median : 27     
##  -0.6992:    2                               Mean   : 24     
##  -0.0113:    1                               3rd Qu.: 46     
##  -0.0131:    1                               Max.   : 90     
##  (Other):  313                               NA's   :19216   
##  max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
##  Min.   :-151             :19216   Min.   :-72      Min.   :-180     
##  1st Qu.:   0      #DIV/0!:   84   1st Qu.: -6      1st Qu.:-175     
##  Median : 113      -1.2   :   32   Median :  0      Median : -61     
##  Mean   :  81      -1.3   :   31   Mean   :  0      Mean   : -58     
##  3rd Qu.: 175      -1.4   :   24   3rd Qu.: 12      3rd Qu.:   0     
##  Max.   : 180      -1.5   :   24   Max.   : 62      Max.   : 167     
##  NA's   :19216     (Other):  211   NA's   :19216    NA's   :19216    
##  min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
##         :19216   Min.   :  0            Min.   :  0            
##  #DIV/0!:   84   1st Qu.:  1            1st Qu.:  2            
##  -1.2   :   32   Median : 18            Median : 84            
##  -1.3   :   31   Mean   : 25            Mean   :139            
##  -1.4   :   24   3rd Qu.: 40            3rd Qu.:350            
##  -1.5   :   24   Max.   :126            Max.   :360            
##  (Other):  211   NA's   :19216          NA's   :19216          
##  amplitude_yaw_forearm total_accel_forearm var_accel_forearm
##         :19216         Min.   :  0.0       Min.   :  0      
##  #DIV/0!:   84         1st Qu.: 29.0       1st Qu.:  7      
##  0      :  322         Median : 36.0       Median : 21      
##                        Mean   : 34.7       Mean   : 34      
##                        3rd Qu.: 41.0       3rd Qu.: 51      
##                        Max.   :108.0       Max.   :173      
##                                            NA's   :19216    
##  avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
##  Min.   :-177     Min.   :  0         Min.   :    0    Min.   :-68      
##  1st Qu.:  -1     1st Qu.:  0         1st Qu.:    0    1st Qu.:  0      
##  Median :  11     Median :  8         Median :   64    Median : 12      
##  Mean   :  33     Mean   : 42         Mean   : 5274    Mean   : 12      
##  3rd Qu.: 107     3rd Qu.: 85         3rd Qu.: 7289    3rd Qu.: 28      
##  Max.   : 177     Max.   :179         Max.   :32102    Max.   : 72      
##  NA's   :19216    NA's   :19216       NA's   :19216    NA's   :19216    
##  stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm stddev_yaw_forearm
##  Min.   : 0           Min.   :   0      Min.   :-155    Min.   :  0       
##  1st Qu.: 0           1st Qu.:   0      1st Qu.: -26    1st Qu.:  1       
##  Median : 6           Median :  30      Median :   0    Median : 25       
##  Mean   : 8           Mean   : 140      Mean   :  18    Mean   : 45       
##  3rd Qu.:13           3rd Qu.: 166      3rd Qu.:  86    3rd Qu.: 86       
##  Max.   :48           Max.   :2280      Max.   : 169    Max.   :198       
##  NA's   :19216        NA's   :19216     NA's   :19216   NA's   :19216     
##  var_yaw_forearm gyros_forearm_x   gyros_forearm_y  gyros_forearm_z 
##  Min.   :    0   Min.   :-22.000   Min.   : -7.02   Min.   : -8.09  
##  1st Qu.:    0   1st Qu.: -0.220   1st Qu.: -1.46   1st Qu.: -0.18  
##  Median :  612   Median :  0.050   Median :  0.03   Median :  0.08  
##  Mean   : 4640   Mean   :  0.158   Mean   :  0.08   Mean   :  0.15  
##  3rd Qu.: 7368   3rd Qu.:  0.560   3rd Qu.:  1.62   3rd Qu.:  0.49  
##  Max.   :39009   Max.   :  3.970   Max.   :311.00   Max.   :231.00  
##  NA's   :19216                                                      
##  accel_forearm_x  accel_forearm_y accel_forearm_z  magnet_forearm_x
##  Min.   :-498.0   Min.   :-632    Min.   :-446.0   Min.   :-1280   
##  1st Qu.:-178.0   1st Qu.:  57    1st Qu.:-182.0   1st Qu.: -616   
##  Median : -57.0   Median : 201    Median : -39.0   Median : -378   
##  Mean   : -61.7   Mean   : 164    Mean   : -55.3   Mean   : -313   
##  3rd Qu.:  76.0   3rd Qu.: 312    3rd Qu.:  26.0   3rd Qu.:  -73   
##  Max.   : 477.0   Max.   : 923    Max.   : 291.0   Max.   :  672   
##                                                                    
##  magnet_forearm_y magnet_forearm_z
##  Min.   :-896     Min.   :-973    
##  1st Qu.:   2     1st Qu.: 191    
##  Median : 591     Median : 511    
##  Mean   : 380     Mean   : 394    
##  3rd Qu.: 737     3rd Qu.: 653    
##  Max.   :1480     Max.   :1090    
## 
```

Again, we were left with 13 sensors out of 38. The list of sensor that will be considered in the predictive model is given below.


```r
valid_forearm_predictors <- c(122:124,140, 151:159)
names(ds[,valid_forearm_predictors])
```

```
##  [1] "roll_forearm"        "pitch_forearm"       "yaw_forearm"        
##  [4] "total_accel_forearm" "gyros_forearm_x"     "gyros_forearm_y"    
##  [7] "gyros_forearm_z"     "accel_forearm_x"     "accel_forearm_y"    
## [10] "accel_forearm_z"     "magnet_forearm_x"    "magnet_forearm_y"   
## [13] "magnet_forearm_z"
```

## Training and testing dataset

The model we will be building will have 52 predictors. Now that the needed predictors have been identified we can reduce the dataset initially loaded by eliminating the unwanted predictor.


```r
valid_predictor <- c(valid_belt_predictors, valid_arm_predictors, valid_dumbbell_predictors, valid_forearm_predictors)
reduced_ds <- ds[,c(valid_predictor,160)]
```

We can also generate the training and testing datasets that will respectively be use to build and validate our predictive model.



```r
set.seed(13579)

inTrain <- createDataPartition(y=reduced_ds$classe,p=0.70, list=FALSE)
training_reduced_ds <- reduced_ds[inTrain,]
testing_reduced_ds <- reduced_ds[-inTrain,]

dim(training_reduced_ds)
```

```
## [1] 13737    53
```

```r
dim(testing_reduced_ds)
```

```
## [1] 5885   53
```

## Model development

In this section we are going to used the training dataset described above to build a 
predictive model base on the random forrest method, This method has been chosen as it was the one favoured in the study on which this project is based.

To build the random forrest model we have chosen to use the randomForest() function rather that the train function for performance reasons. THe randomForrest() function took only a few minutes to generate the model when it took a few hours to do the same task using the train() function with the parameter "rf"


```r
modelfit <- randomForest(classe ~ .,data=training_reduced_ds)
modelfit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training_reduced_ds) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.53%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3904    2    0    0    0    0.000512
## B   14 2636    8    0    0    0.008277
## C    0   10 2385    1    0    0.004591
## D    0    0   28 2222    2    0.013321
## E    0    0    3    5 2517    0.003168
```

We run the model on the test data to validate its accuracy


```r
predictions <- predict(modelfit, newdata = testing_reduced_ds)
confusionMatrix(predictions,testing_reduced_ds$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    8    0    0    0
##          B    1 1126    6    0    0
##          C    0    5 1018    8    1
##          D    0    0    2  956    3
##          E    1    0    0    0 1078
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.992, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.992         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.989    0.992    0.992    0.996
## Specificity             0.998    0.999    0.997    0.999    1.000
## Pos Pred Value          0.995    0.994    0.986    0.995    0.999
## Neg Pred Value          1.000    0.997    0.998    0.998    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.173    0.162    0.183
## Detection Prevalence    0.285    0.193    0.175    0.163    0.183
## Balanced Accuracy       0.998    0.994    0.995    0.995    0.998
```

We can see that the model developed has a very high level of accuracy at 99.58%. In the next section we are going to verify the this high level of accuracy is related to the quality of the model and is not the result of overfitting the model to the training dataset.


##Error estimation Using K-fold

To confirm the validity of the error estimation for our model above  and to verify we have not overfitted our model we are going to do a cross validation using k-fold. According to the literature on the subject, 10-fold cross-validation is commonly used for this purpose and will be the approach chosen and described below  

We first generat our 10 mutually exclusive folds


```r
set.seed(2468)

folds <- createFolds(y=reduced_ds$classe, k=10, list = TRUE, returnTrain=FALSE)
sapply(folds,length)
```

```
## Fold01 Fold02 Fold03 Fold04 Fold05 Fold06 Fold07 Fold08 Fold09 Fold10 
##   1962   1963   1962   1963   1961   1962   1963   1963   1961   1962
```

We create 10 training and testing datasets based on the folds generate above.


```r
training_reduced_ds_kf01 <- training_reduced_ds[-folds$Fold01,]
testing_reduced_ds_kf01 <- training_reduced_ds[folds$Fold01,]

training_reduced_ds_kf02 <- training_reduced_ds[-folds$Fold02,]
testing_reduced_ds_kf02 <- training_reduced_ds[folds$Fold02,]

training_reduced_ds_kf03 <- training_reduced_ds[-folds$Fold03,]
testing_reduced_ds_kf03 <- training_reduced_ds[folds$Fold03,]

training_reduced_ds_kf04 <- training_reduced_ds[-folds$Fold04,]
testing_reduced_ds_kf04 <- training_reduced_ds[folds$Fold04,]

training_reduced_ds_kf05 <- training_reduced_ds[-folds$Fold05,]
testing_reduced_ds_kf05 <-  training_reduced_ds[folds$Fold05,]

training_reduced_ds_kf06 <- training_reduced_ds[-folds$Fold06,]
testing_reduced_ds_kf06 <- training_reduced_ds[folds$Fold06,]

training_reduced_ds_kf07 <- training_reduced_ds[-folds$Fold07,]
testing_reduced_ds_kf07 <- training_reduced_ds[folds$Fold07,]

training_reduced_ds_kf08 <- training_reduced_ds[-folds$Fold08,]
testing_reduced_ds_kf08 <- training_reduced_ds[folds$Fold08,]

training_reduced_ds_kf09 <- training_reduced_ds[-folds$Fold09,]
testing_reduced_ds_kf09 <- training_reduced_ds[folds$Fold09,]

training_reduced_ds_kf10 <- training_reduced_ds[-folds$Fold10,]
testing_reduced_ds_kf10 <- training_reduced_ds[folds$Fold10,]
```

We create 10 predictive models using the training datasets generated above 


```r
modelfit_kf01 <- randomForest(classe~ .,data=training_reduced_ds_kf01)
modelfit_kf02 <- randomForest(classe~ .,data=training_reduced_ds_kf02)
modelfit_kf03 <- randomForest(classe~ .,data=training_reduced_ds_kf03)
modelfit_kf04 <- randomForest(classe~ .,data=training_reduced_ds_kf04)
modelfit_kf05 <- randomForest(classe~ .,data=training_reduced_ds_kf05)
modelfit_kf06 <- randomForest(classe~ .,data=training_reduced_ds_kf06)
modelfit_kf07 <- randomForest(classe~ .,data=training_reduced_ds_kf07)
modelfit_kf08 <- randomForest(classe~ .,data=training_reduced_ds_kf08)
modelfit_kf09 <- randomForest(classe~ .,data=training_reduced_ds_kf09)
modelfit_kf10 <- randomForest(classe~ .,data=training_reduced_ds_kf10)
```

We predict for each models using the 10  testing datasets initially created
using our 10 mutually exclusive folds.


```r
predictions_kf01 <- predict(modelfit_kf01, newdata = testing_reduced_ds_kf01)
predictions_kf02 <- predict(modelfit_kf02, newdata = testing_reduced_ds_kf02)
predictions_kf03 <- predict(modelfit_kf03, newdata = testing_reduced_ds_kf03)
predictions_kf04 <- predict(modelfit_kf04, newdata = testing_reduced_ds_kf04)
predictions_kf05 <- predict(modelfit_kf05, newdata = testing_reduced_ds_kf05)
predictions_kf06 <- predict(modelfit_kf06, newdata = testing_reduced_ds_kf06)
predictions_kf07 <- predict(modelfit_kf07, newdata = testing_reduced_ds_kf07)
predictions_kf08 <- predict(modelfit_kf08, newdata = testing_reduced_ds_kf08)
predictions_kf09 <- predict(modelfit_kf09, newdata = testing_reduced_ds_kf09)
predictions_kf10 <- predict(modelfit_kf10, newdata = testing_reduced_ds_kf10)
```

For each model we evaluate a confusion matrix


```r
cm_kf01 <- confusionMatrix(predictions_kf01,testing_reduced_ds_kf01$classe)
cm_kf02 <- confusionMatrix(predictions_kf02,testing_reduced_ds_kf02$classe)
cm_kf03 <- confusionMatrix(predictions_kf03,testing_reduced_ds_kf03$classe)
cm_kf04 <- confusionMatrix(predictions_kf04,testing_reduced_ds_kf04$classe)
cm_kf05 <- confusionMatrix(predictions_kf05,testing_reduced_ds_kf05$classe)
cm_kf06 <- confusionMatrix(predictions_kf06,testing_reduced_ds_kf06$classe)
cm_kf07 <- confusionMatrix(predictions_kf07,testing_reduced_ds_kf07$classe)
cm_kf08 <- confusionMatrix(predictions_kf08,testing_reduced_ds_kf08$classe)
cm_kf09 <- confusionMatrix(predictions_kf09,testing_reduced_ds_kf09$classe)
cm_kf10 <- confusionMatrix(predictions_kf10,testing_reduced_ds_kf10$classe)
```

We see that for each model we have a very high level of accurracy


```r
cm_kf01$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9913         0.9890         0.9849         0.9955         0.2797 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf02$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9957         0.9945         0.9906         0.9984         0.2945 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf03$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9963         0.9954         0.9915         0.9988         0.2867 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf04$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9920         0.9899         0.9858         0.9960         0.2883 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf05$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9956         0.9944         0.9905         0.9984         0.2952 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf06$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9949         0.9936         0.9896         0.9980         0.2750 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf07$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9949         0.9936         0.9895         0.9980         0.2698 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf08$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9942         0.9926         0.9885         0.9975         0.2863 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf09$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9912         0.9889         0.9847         0.9954         0.2839 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

```r
cm_kf10$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##         0.9964         0.9954         0.9916         0.9988         0.2841 
## AccuracyPValue  McnemarPValue 
##         0.0000            NaN
```

We calculate the average error rate of all 10 models


```r
Average_error <- (cm_kf01$overall[1]+cm_kf02$overall[1]+cm_kf03$overall[1]+
cm_kf04$overall[1]+cm_kf05$overall[1]+cm_kf06$overall[1]+
cm_kf07$overall[1]+cm_kf08$overall[1]+cm_kf09$overall[1]+
cm_kf10$overall[1])/10

Average_error
```

```
## Accuracy 
##   0.9942
```
We can see that the error rate calculate using the K-fold cross validation is virtually identical to the error previously evaluated. We can not then conclude that there was some overfitting in the initial model. The out of sample error will be higher than the error calculated here but we don't expect it to be a lot higher as the model seems to efficently hignore the noise present in the signal.


## Test Validation


```r
test_ds<-read.csv("pml-testing.csv")
reduced_test_ds <- test_ds[,c(valid_predictor,160)]

test_predictions <- predict(modelfit, newdata = reduced_test_ds)
test_predictions
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
