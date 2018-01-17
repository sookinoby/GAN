In our previous notebooks, we used a deep learning technique called Convolution Neural Network (CNN) to classify text and images.  A CNN is an example of a Discriminative Model, which creates a  decision boundary to classify a given input signal (data).

Deep learning models in recent times have been used to create even more powerful and useful models called Generative Models. A Generative Model doesn’t just create a decision boundary but understands the underlying distribution of values. Using this insight, a generative model can also generate new data or classify a given input data. Here are some examples of Generative Models:

 1.Producing a new song or combine two genres of songs to create an entirely different song, and synthesizing new images from existing images are some examples of Generative Models.

2. [Upgrading images](https://arxiv.org/pdf/1703.04244.pdf) to a higher resolution for removing fuzziness, improving image quality, restoring photos, and much more.


In general, Generative Models can be used on any form of data to learn the underlying distribution, generate new data, and augment existing data.

In this tutorial, we are going to build Generative Models, specially generative adversarial network (GAN) for generating a new image from existing images. using Apache MXNet gluon API.

By the end of the notebook, you will be able to:

1. Understand Generative Models
2. Understand Generative Models in context of deep neural network
3. Implement a Generative Adversarial Network (GAN)

## How Generative Models Go Further Than Discriminative Models

Let’s understand the power of Generative Models using a trivial example.

The following table depicts the heights of ten humans and Martians.  

Martian (height in centimetre) - 250,260,270,300,220,260,280,290,300,310 <br />
Human (height in centimetre) - 160,170,180,190,175,140,180,210,140,200 <br />


The heights of human beings follow a normal distribution, showing up as a bell-shaped curve on the graph. Martians tend to be much taller than humans but also have a normal distribution. So let's input the heights of humans and Martians into both Discriminative and Generative models. 


If we train a Discriminative Model, it will only plot a decision boundary. The model misclassifies just one human - the accuracy is quite good overall. Basically, the model doesn’t learn about the underlying distribution of data so it is not suitable to build powerful applications listed in the beginning of this article. ![Alt text](images/martians-chart5_preview.jpeg?raw=true "Unrolled RNN") <br />

In contrast, a generative model will learn the underlying distribution (lower dimension representation) for Martian (mean =274, std= 8.71) and Human (mean=174, std=7.32).  ![Alt text](images/humans_mars.png?raw=true "Unrolled RNN")<br />. Suppose we have a normal distribution for Martian (mean =274, std = 8.71), we can produce new data by generating a random number between 0 and 1 (uniform distribution) and then querying the normal distribution of Martians to get a value say 275 cm.

Using the underlying distribution, we can generate new Martians and Humans, or a new interbreed species (humars). We have the infinite ways to generate data as we can manipulate the underlying distribution of data.  We can also use this model for classifying Martians and Humans, just like the discriminative model. For a concrete understanding of generative vs discriminative models, please check [this](https://arxiv.org/pdf/1703.01898.pdf).

Examples of Discriminative models - Logistic regression, Support Vector Machine, etc.
Examples of Generative models -Hidden Markov Model, Naive Bayes Classifier, etc.

### Generative vs Discriminative Models in neural network

Let’s say you want to train two models called “m-dis” and “m-gen-partial” to find the difference between a dog and a cat. 
 
An “m-dis” will have a [softmax layer](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss) at the end (final layer), which does binary classification.  All the other layers (hidden layer) try to learn a [representation](http://www.deeplearningbook.org/contents/representation.html) of the input (cat/dog) that can reduce the loss at the final layer. The hidden layer may* learn a rule like : <br />
If the eyes are blue and have brown strips then it is a cat or it is a dog, ignoring other important features like the shape of the body, height, etc.
    
On the other hand, “m-gen-partial” is trained to learn a lower dimension representation (distribution) that can represent the input image of cat/dog. The final layer is not a softmax layer used for classification. The hidden layer can learn about the general features of a cat/dog (shape, colour, height, etc). Moreover, the dataset needs no labelling as we are only training to extract features to represent the input data. Then we can tweak the model “‘m-gen-partial’” to classify a cat/dog by adding a softmax classifier at the end and by training with few labelled examples of cat/dog. We can also generate new data by adding a decoder network to the ‘m-gen-partial’ model. Adding a decoder network is not trivial -- we have explained about this in the “GAN model” section.
 
* - In a deep neural network, the hidden layers of the discriminative model actually learns the general features except for the last layer which is used in classification.  

### Preparing your environment

If you're working in the AWS Cloud, you can save yourself a lot of installation work by using an [Amazon Machine Image](https://aws.amazon.com/marketplace/pp/B01M0AXXQB#support), pre-configured for deep learning.  If you have done this, skip steps 1-5 below.

If you are using a Conda environment, remember to install pip inside conda by typing 'conda install pip' after you activate an environment.  This will save you a lot of problems down the road.

Here's how to get set up:

1. Install [Anaconda](https://www.continuum.io/downloads), a package manager. It is easier to install Python libraries using Anaconda.
2. Install [scikit-learn](http://scikit-learn.org/stable/install.html), a general-purpose scientific computing library. We'll use this to pre-process our data. You can install it with 'conda install scikit-learn'.
3. Grab the Jupyter Notebook, with 'conda install jupyter notebook'.
4. Get [MXNet](https://github.com/apache/incubator-mxnet/releases), an open source deep learning library. The Python notebook was tested on version 0.12.0 of MxNet, and  you can install using pip as follows: pip install mxnet==0.12.0
5. After you activate the anaconda environment, type these commands in it: ‘source activate mxnet’

The consolidated list of commands are given below
```bash
conda install pip
pip install opencv-python
conda install scikit-learn
conda install jupyter notebook
pip install mxnet==0.12.0
```

6. You can download the MXNet notebook for this part of the tutorial [here](https://github.com/sookinoby/generative-models/blob/master/Test-rnn.ipynb), where we've created and run all this code, and play with it! Adjust the hyperparameters and experiment with different approaches to neural network architecture.

## Generative Adversarial Network (GAN)

[Generative Adversarial Network](https://arxiv.org/abs/1406.2661) is a neural network model based on a [zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game) from game theory. The application typically consists of two different neural networks called Discriminator and Generator, where each network tries to outperform the other. Let us consider an example to understand GAN network.

Let’s assume that there is a bank (discriminator) that detects whether a given currency is real or fake using machine learning. A fraudster (generator) builds a machine learning model to counterfeit fake currency notes by looking at the real currency notes and deposits them in the bank. The bank tries to identify the currencies deposited as fake.
![Alt text](images/GAN_SAMPLE.png?raw=true "Generative Adversarial Network")

If the bank tells the fraudster why it classified these notes as fake,  he can improve his model based on those reasons. After multiple iterations, the bank cannot find the difference between the “real” and “fake” currency. This is the idea behind GAN. So now let's implement a simple GAN network.

I encourage you to download [the notebook](https://github.com/sookinoby/generative-models/blob/master/GAN.ipynb).
You are welcome to adjust the hyperparameters and experiment with different approaches to neural network architecture.

### Preparing the DataSet

We use a library called [Brine](https://docs.brine.io/getting_started.html) to download our dataset. Brine has many data sets, so we can choose the data set that we want to download. To install Brine and download our data set, do the following:

1. pip install brine-io
2. brine install jayleicn/anime-faces

For this tutorial, I am using the Anime-faces dataset, which contains over 100,000 anime images collected from the Internet.

Once the dataset is downloaded, you can load it using the following code:

```python
# brine for loading anime-faces dataset
import brine
anime_train = brine.load_dataset('jayleicn/anime-faces')
```


We also need to normalize the pixel value of each image to [-1 to 1] and reshape each image from (width X height X channels) to (channels X width X height), because the latter format is what MxNet expects. The transform function does the job of reshaping the input image into the required shape expected by the MxNet model.


```python
def transform(data, target_wd, target_ht):
    # resize to target_wd * target_ht
    data = mx.image.imresize(data, target_wd, target_ht)
    # transpose from (target_wd, target_ht, 3)
    # to (3, target_wd, target_ht)
    data = nd.transpose(data, (2,0,1))
    # normalize to [-1, 1]
    data = data.astype(np.float32)/127.5 - 1
    return data.reshape((1,) + data.shape)
```
The getImageList function reads the images from the training_folder and returns the images as a list, which is then transformed into a MxNet array.

```python
# Read images, call the transform function, attach it to list
def getImageList(base_path,training_folder):
    img_list = []
    for train in training_folder:
        fname = base_path + train.image
        img_arr = mx.image.imread(fname)
        img_arr = transform(img_arr, target_wd, target_ht)
        img_list.append(img_arr)
    return img_list
```
base_path = 'brine_datasets/jayleicn/anime-faces/images/'
img_list = getImageList('brine_datasets/jayleicn/anime-faces/images/',training_fold)
```


### Designing the network

We now need to design the two separate networks, the discriminator and the generator. The generator takes a random vector of shape (batchsize X N ), where N is an integer and converts it to an image of shape (batch size X channels X width X height). 

It uses [transpose convolutions](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#no-zero-padding-unit-strides-transposed) to upscale the input vectors. 
This is very similar to how a decoder unit in an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) maps a lower-dimension vector into a higher-dimensional vector representation. You can choose to design your own generator network, the only the thing you need to be careful about is the input and the output shapes. The input to generator network should be of low dimension (we use 1X150 dimension, latent_z_size) and output should be the expected number of channels (3, for color images), width and height (3 x width x height). Here’s the snippet of a generator network.


```python

# Simple generator. You can use any model of your choice(VGG, AlexNet, etc.) but ensure that it upscales the latent variable(random vectors) to 64 * 64 * 3 channel image - the output image we want the generative model to produce.
With netG.name_scope():
     # input is random_z (batchsize X 150 X 1), going into a tranposed convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # output size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # output size. (ngf*8) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # output size. (ngf*8) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # output size. (ngf*8) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1))
    netG.add(nn.Activation('tanh')) # use tanh , we need an output that is between -1 to 1, not 0 to 1 
    # Remember the input image is normalised between -1 to 1, so should be the output
    # output size. (nc) x 64 x 64
```

Our discriminator is a binary image classification network that maps the image of shape (batch size X channels X width x height) into a lower-dimension vector of shape (batch size X 1). This is similar to an encoder that converts a higher-dimension image representation into a lower-dimension one. Again, you can use any model that does binary classification with reasonable accuracy. 

Here’s the snippet of the discriminator network:

```python
with netD.name_scope():
    # input is (nc) x 64 x 64
    netD.add(nn.Conv2D(ndf, 4, 2, 1))
    netD.add(nn.LeakyReLU(0.2))
    # output size. (ndf) x 32 x 32
    netD.add(nn.Conv2D(ndf * 2, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # output size. (ndf) x 16 x 16
    netD.add(nn.Conv2D(ndf * 4, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # output size. (ndf) x 8 x 8
    netD.add(nn.Conv2D(ndf * 8, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # output size. (ndf) x 4 x 4
    netD.add(nn.Conv2D(1, 4, 1, 0))
```

## Training the GAN network

The training of a GAN network is not straightforward, but it is simple. The following diagram illustrates the training process.  ![Alt text](images/GAN_Model.png?raw=true "GAN training") <br />

The real images are given a label of 1, and the fake images are given a label of 0.

```python
#real label is the labels of real image
real_label = nd.ones((batch_size,), ctx=ctx)
#fake labels is label associated with fake image
fake_label = nd.zeros((batch_size,),ctx=ctx)
```
### Training the discriminator

A real image is now passed to the discriminator, to determine if it is real or fake, and the loss associated with the prediction is calculated as errD_real.

 ```python
# train with real image
output = netD(data).reshape((-1, 1))
#The loss is a real valued number
errD_real = loss(output, real_label)
```

In the next step, a random noise random_z is passed to the generator network to produce a random image. This image is then passed to the discriminator to classify it as real (1) or fake(0), thereby creating a loss, errD_fake. This errD_fake is high if the discriminator wrongly classifies the fake image (label 0) as a true image (label 1). This errD_fake is back propagated to train the discriminator to classify the fake image as a fake image (label 0). This helps the discriminator to improve its accuracy.

 ```python
#train with fake image, see what the discriminator predicts
#creates fake image
fake = netG(random_z)
# pass it to the discriminator
output = netD(fake.detach()).reshape((-1, 1))
errD_fake = loss(output, fake_label)
 ```

The total error is back propagated to tune the weights of the discriminator.

 ```python
#compute the total error for fake image and the real image
errD = errD_real + errD_fake
#improve the discriminator skill by back propagating the error
errD.backward()
```

### Training the generator

The random noise(random_z) vector used for training the discriminator is used again to generate a fake image. We then pass the fake image to the discriminator network to obtain the classification output, and the loss is calculated. The loss is high if the fake image generated (label = 0) is not similar to the real image (label 1) i.e. The generator is not able to produce a fake image that can trick the discriminator to classify it as a real image (label =1). The loss is then used to fine-tune the generator network.

```python
fake = netG(random_z)
output = netD(fake).reshape((-1, 1))
errG = loss(output, real_label)
errG.backward()
```

### Generating new fake images
The model weights are available [here](https://www.dropbox.com/s/uu45cq5y6uigiro/GAN_t2.params?dl=0). You can download the model parameters and load it using [model.load_params](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=load#mxnet.module.BaseModule.load_params) function.
We can use the generator network to create new fake images by providing 150 random dimensions as an input to the network. 

 ![Alt text](images/GAN_image.png?raw=true "GAN generated images")<br />

```
#Let’s generate some random images
num_image = 8
for i in range(num_image):
    # random input for generating images
    latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
    img = netG(random_z)
    plt.subplot(2,4,i+1)
    visualize(img[0])
plt.show()
```
Although, the images generated looks similar to the input dataset, they look fuzzy. There are several other [GAN]![https://blog.openai.com/generative-models/] network  that you can experiment with and achieve amazing results.

# Conclusion

Generative models open up new opportunities for deep learning. This article has explored some of the famous generative models for image data. We learned about GAN models and generated images identical to the input data (Anime Characters). 