# NAS-SegNet

# THIS IS THE MAIN CODE BRANCH

## Description

**NAS-SegNet** Goal of this project is to implement NASSegNet for nuclei image segmentation with analog/hardware-based neural architecture search (HW-NAS). Until now, only classification architectures have been implemented in this domain, so the unique value of this solution is the new segmentation implementation.

## Code Updates:
[**NAS Run**](https://github.com/vishgoki/nas-seg-net/blob/nas-nuclei/AnalogNAS_Run.ipynb)
[**Digital Training**](https://github.com/vishgoki/nas-seg-net/blob/nas-nuclei/NAS_NUCLEI_RESNETSEG_DIGITAL.ipynb)
[**Analog Training**](https://github.com/vishgoki/nas-seg-net/blob/nas-nuclei/NAS_NUCLEI_RESNETSEG_ANALOG_Latest.ipynb)

## Approach

* Use the MONAI (Medical Open Network for AI) framework for dataset pre-processing and augmentation.
* Adapt the existing IBM Analog-NAS Macro-Architecture which performs image classification by default to a novel Macro-Architecture which is utilized to run a neural architecture search for generating an optimized NASSegNet model architecture for nuclei segmentation.
* The Analog-NAS approach explores different neural network configurations, evaluating their performance on the target task and hardware constraints, to find the most efficient architecture. This is done by using the pretrained surrogate models.
* We then train the model architecture with the best accuracy using digital and analog methods.


## Results

* Successfully implemented the NASSegNet architecture for nuclei segmentation. 
* Leveraged the IBM Analog-NAS tool to perform a neural architecture search, resulting in an optimized NASSegNet model with best accuracy for this task.
* Trained the optimal network generated model (digital and analog training) with the best model using IBM AIHWKIT.
![image](https://github.com/vishgoki/nas-seg-net/assets/57005975/2655ae1a-c31b-460d-98d1-f08953261867)
<img width="683" alt="image" src="https://github.com/vishgoki/nas-seg-net/assets/57005975/4197b2e1-ea12-4d5a-bb42-913d567acc85">


## Technical Challenges
 
AnalogaiNAS package offers the following features: 

* Utilizing BootstrapNAS by Intel to create a search space and macro architecture for segmentation using UNet Architecture – This was tightly coupled with Intel’s NNCF library and doesn’t support integration with AnalogNAS.
* AnalogNAS and its Image classification dependency: AnalogNAS’s search space, macro architecture are suited for image classification.
* Implementing NASSegNet for nuclei segmentation poses significant technical challenges. 

## Dataset and Data Preparation

Nuclear segmentation in digital microscopic tissue images can enable extraction of high-quality features for nuclear morphometric and other analyses in computational pathology. However, conventional image processing techniques such as Otsu and watershed segmentation do not work effectively on challenging cases such as chromatin-sparse and crowded nuclei. In contrast, machine learning-based segmentation techniques are able to generalize over nuclear appearances. 
Finally, data augmentation and preprocessing transforms are applied using train_transforms and val_transforms and supplied to the model via dataloaders.

Before:
<img width="227" alt="image" src="https://github.com/vishgoki/nas-seg-net/assets/57005975/6eb8be9e-e9db-45bc-ba25-101c0360cc3e">
After:
<img width="227" alt="image" src="https://github.com/vishgoki/nas-seg-net/assets/57005975/78cc45bd-504d-4cd7-a074-82adde19b0bc">

## Training the NASSegNet Model with Analog AI NAS

![image](https://github.com/vishgoki/nas-seg-net/assets/57005975/6a7457e0-f5f0-403c-9834-242130bdecdc)
We define the NASSegNet model architecture, specifying parameters like input/output channels and number of units, downsample, upsample/transpose conv layers.

<img width="409" alt="image" src="https://github.com/vishgoki/nas-seg-net/assets/57005975/aed216d0-91f5-409f-b422-39aff4414d0b">
Set up the Dice Loss as the loss function and the Dice Metric as the evaluation metric for training and validating the model.
![image](https://github.com/vishgoki/nas-seg-net/assets/57005975/2d4e10be-7de0-46d3-be6f-282482ca8fb4)

## Observations and Conclusion

* Successfully implemented the NASSegNet architecture for nuclei segmentation. Generated architecture of model with best accuracy which was then trained digitally and also analog training was performed with aihwtoolkit.
* Leveraged the IBM Analog-NAS tool to perform a neural architecture search. Also worked with AIHWToolkit for analog training.
* This solution represents a implementation of NASSegNet for medical image segmentation, going beyond the previous use of ResNet-like architectures in this domain.

## References
* [IBM/analog-nas]([https://www.ijcai.org/proceedings/2021/592](https://github.com/IBM/analog-nas))
* [AIHWKit](https://ieeexplore.ieee.org/abstract/document/9458494)

## License
This project is licensed under [Apache License 2.0].

[Apache License 2.0]: LICENSE.txt
