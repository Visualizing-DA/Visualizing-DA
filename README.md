# Vsualizing-Transferred-Knowledge
Pytorch implementation for paper **Visualizing Transferred Knowledge: An Interpretive Model of Unsupervised Domain Adaptation**. The full code will be released shortly.

# Abstract
Many research efforts have been committed to unsupervised domain adaptation (DA) problems that transfer knowledge learned from a labeled source domain to an unlabeled target domain. Various DA methods have achieved remarkable results recently in terms of predicting ability, which implies the effectiveness of the aforementioned knowledge transferring. However, state-of-the-art methods rarely probe deeper into the transferred mechanism, leaving the true essence of such knowledge obscure. Recognizing its importance in the adaptation process, we propose an interpretive model of unsupervised domain adaptation, as the first attempt to visually unveil the mystery of transferred knowledge. Adapting the existing concept of the prototype from visual image interpretation to the DA task, our model similarly extracts shared information from the domain-invariant representations as prototype vectors. Furthermore, we extend the current prototype method with our novel prediction calibration and knowledge fidelity preservation modules, to orientate the learned prototypes to the actual transferred knowledge. By visualizing these prototypes, our method not only provides an intuitive explanation for the base model's predictions but also unveils transfer knowledge by matching  the image patches with the same semantics across both source and target domains. Comprehensive experiments and in-depth explorations demonstrate the efficacy of our method in understanding the transferred mechanism and its potential in downstream tasks including model diagnosis. 

# Prerequisites
- python >= 3.6.8
- pytorch ==>=1.7.0
- torchvision == >=0.5.0
- numpy, scipy, PIL, argparse, tqdm, pandas,prettytable,scikit-learn,webcolors,matplotlib,opencv-python,numba

# Framework
![Alt text](framework.png?raw=true "Title")


# Base DA methods
We run our base DA methods based on the implementation of [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library).
We use the default setting the in their [example codes](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/image_classification) to run DANN.
We added the NWD core code from [DALN](https://github.com/xiaoachen98/DALN) to the [MCC](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/mcc.py) code for NWD+MCC base model. 


# Datasets
Please download [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view) and [DomainNet](http://ai.bu.edu/M3SDA/) datasets.
Organize the datasets as following:

    ---office-home
       |--- Art
          |--- Alarm_Clock
          ...
          |--- Webcam
       |--- Clipart
          ...
       |--- Product
          ...
       |--- Real_World


# Running the experiments
1. Follow the [installation instrution](https://github.com/thuml/Transfer-Learning-Library#Installation) of tll liabrary and train the base model with tll.
2. For NWD+MCC model, add the core code of NWD loss from [DALN](https://github.com/xiaoachen98/DALN) to the [MCC](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/mcc.py).
3. The pre-trained DANN checkpoint for Ar->Pr on Office-Home dataset can be download [here](https://drive.google.com/file/d/13nGzWd23wuSTi3Awzo0xQKBY2dwKcjxe/view?usp=sharing).
4. Modify the setting file and train our proposed model:
  > python main_reload_nonml_dann.py -gpuid 0


# Acknowledgement
This project is built on the open-source implementation [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) and [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library). Thank the authors for their excellent work.
