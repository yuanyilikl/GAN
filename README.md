# GAN
Deep learning the GAN  
Learn to generate the model through MNIST data, Anime data and other picture data.  
The models mainly include: VAE, GAN, CGAN , DCGAN.
## Environment

* numpy
* torch
* torchvision
* tqdm
* fire

## Usage

Examples of Usage:

* **Train:**  
 mnist_dcgan.py  

```
    python mnist_dcgan.py train  # If gpu is used
    python mnist_dcgan.py train --use_gpu=False  # If cpu is used
```
If your train is interrupted, you can continue training with "gen_epoch"
```
    python mnist_dcgan.py train --gen_epoch=3  
```
* **Generate:**  
mnist_dcgan.py
```
    python mnist_dcgan.py generate --use_gpu=False #If cpu is used
```
   You can use "digit" to generate the number you want
``` 
    python mnist_dcgan.py generate --gen_epoch=3 --digit=7 
```
## Reference
[chenyun](https://github.com/chenyuntc/pytorch-book)

[DeepLearningTutorial](https://github.com/Flowingsun007/DeepLearningTutorial)

[python技术交流与分享](http://www.feiguyunai.com/)

[The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)

[Andrew Ng coursera](https://zh.coursera.org/learn/machine-learning)，
[Andrew Ng bilibili](https://www.bilibili.com/video/BV164411m79z?p=17).