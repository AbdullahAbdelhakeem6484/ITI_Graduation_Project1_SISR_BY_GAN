<p  align="center">
<img src="https://user-images.githubusercontent.com/45875057/147377970-41846090-aeb3-4600-97a3-85566f94f982.png">
</p>


<h1>SRGAN</h1>
<p> was proposed by researchers at Twitter. The motive of this architecture is to recover finer textures from the image when we upscale it so that it’s quality cannot be compromised. There are other methods such as Bilinear Interpolation that can be used to perform this task but they suffer from image information loss and smoothing. In this paper, the authors proposed two architectures the one without GAN (SRResNet) and one with GAN (SRGAN). It is concluded that SRGAN has better accuracy and generate image more pleasing to eyes as compared to SRGAN.</p>
 
<h1>Architecture</h1>
<p>: Similar to GAN architectures, the Super Resolution GAN also contains two parts Generator and Discriminator where generator produces some data based on the probability distribution and discriminator tries to guess weather data coming from input dataset or generator.  Generator than tries to optimize the generated data so that it can fool the discriminator. Below are the generator and discriminator architectural details:</p>
 <p align="center"><img src="https://media.geeksforgeeks.org/wp-content/uploads/20200619230513/SRGAN.jpg"></p>
 
 
 <h1>Generator Architecture:</h1>

<p>The generator architecture contains residual network instead of deep convolution networks because residual networks are easy to train and allows them to be substantially deeper in order to generate better results. This is because the residual network used a type of connections called skip connections.</p>
<p align="center">
<img src="https://www.researchgate.net/profile/Minjun-Li-3/publication/319187018/figure/fig2/AS:529633181667329@1503285798147/Generator-Architecture.png"></p>

<p>There are B residual blocks (16), originated by ResNet. Within the residual block, two convolutional layers are used, with small 3×3 kernels and 64 feature maps followed by batch-normalization layers and ParametricReLU as the activation function.

The resolution of the input image is increased with two trained sub-pixel convolution layers.

This generator architecture also uses parametric ReLU as an activation function which instead of using a fixed value for a parameter of the rectifier (alpha) like LeakyReLU. It adaptively learns the parameters of rectifier and   improves the accuracy at negligible extra computational cost

  During the training, A high-resolution image (HR) is downsampled to a low-resolution image (LR). The generator architecture than tries to upsample the image from low resolution to super-resolution. After then the image is passed into the discriminator, the discriminator and tries to distinguish between a super-resolution and High-Resolution image and generate the adversarial loss which then backpropagated into the generator architecture.</p>
 
 <h1>Discriminator Architecture: </h1>

<p>The task of the discriminator is to discriminate between real HR images and generated SR images.   The discriminator architecture used in this paper is similar to DC- GAN architecture with LeakyReLU as activation. The network contains eight convolutional layers with of 3×3 filter kernels, increasing by a factor of 2 from 64 to 512 kernels. Strided convolutions are used to reduce the image resolution each time the number of features is doubled. The resulting 512 feature maps are followed by two dense layers and a leakyReLU applied between and a final sigmoid activation function to obtain a probability for sample classification.</p> 
 <p align="center">
 <img src="https://user-images.githubusercontent.com/45875057/147367593-46f68e54-8dc6-4c3d-ae8f-d94bb682b621.png"></p>

 <h1>Loss Function:</h1>

<p>The SRGAN uses perpectual loss function (LSR)  which is the weighted sum of two loss components : content loss and adversarial loss. This loss is very important for the performance of the generator architecture:</p>

<ul><li><b>Content Loss:</b> We use two types of content loss in this paper : pixelwise MSE loss for the SRResnet architecture, which is most common MSE loss for image Super Resolution. However MSE loss does not able to deal with high frequency content in the image that resulted in producing overly smooth images. Therefore the authors of the paper decided to  use loss of different VGG layers. This VGG loss is based on the ReLU activation layers of the pre-trained 19 layer VGG network. This loss is defined as follows:</li></ul> 
 <p align="center">
 <img style="text-align:center;" src="https://media.geeksforgeeks.org/wp-content/uploads/20200611204717/simplecontentloss.PNG">
 <img style="text-align:center;" src="https://user-images.githubusercontent.com/45875057/147368424-329fa6d9-d787-499d-b1a8-77efe6830567.png"></p>

 <ul><li><b>Adversarial Loss:</b> The Adversarial loss is the loss function that forces the generator to image more similar to high resolution image by using a discriminator that is trained to differentiate between high resolution and super resolution images.</li></ul>

<p align="center">
<img width="40%" height="40%" src="https://user-images.githubusercontent.com/45875057/147381195-c94bc13a-3871-4017-a928-77bd39cfa910.png"
</p>









<h1 color="green"><b>Results</b></h1>

<p align="center">
 <img src="images/result.PNG">
 
  

<h1 color="green"><b>Inference</b></h1>
<p>Run the infer.py script and pass the required arguments (modelPath, inputPath, outputPath) <br>

python -m infer \ <br>
--modelPath="model.h5" \ <br>
--inputPath="lr_image_path" `Or Image Url ` \ <br>
--outputPath="sr_image_path" \ <br>





<h1 color="green"><b>Instructions to Install our SRGAN Package</b></h1>
<p>Our Package can be found in this link.
 <a href="https://pypi.org/project/SuperResolution-GANs/0.0.3/">https://pypi.org/project/SuperResolution-GANs/0.0.3/</a></p>

1. Install:

```python
pip install SuperResolution-GANs==0.0.3
```
2. Download Our Model:

```python
import gdown
url = 'https://drive.google.com/uc?id=1MWDeLnpEaZDrKK-OjmzvYLxfjwp-GDcp'
output = 'model.h5'
gdown.download(url, output, quiet=False)

```

3. Generate Super Resolution Image:

```python
from super_resolution_gans import srgan_utils
LR_img,SR_img = srgan_utils.SRGAN_predict(lr_image_path"image.jpg", model_path="model.h5")
```
4. show Super Resolution image:

```python
show_image(SR_img)
```






