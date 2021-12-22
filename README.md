<center align="center">
<h1 align="center"><font size="+4"> Single Image Super Resolution by GAN (SRGAN)</font></h1>
</center>

---

 
 


<h1 color="green"><b>Abstract</b></h1>
<p></p>

<h1 color="green"><b>Results</b></h1>
<table style="width:100%">
  <tr>
    <th>Model</th>
    <th>PSNR</th>
    <th>SSIM</th>
    <th>Batch Size</th>
    <th>Image<th>
  </tr>
  
  <tr>
    <td></td> <td></td>  <td></td>  <td></td>  <td></td> <td><img src="" ></td>
  </tr>
 
 
  
</table>

<h1 color="green"><b>Inference</b></h1>
<p>Run the infer.py script and pass the required arguments (modelPath, streaming, inputPath, outputPath, sequenceLength, skip, showInfo) <br>

python -m infer \ <br>
--modelPath="./FDSC/models/model_16_m3_0.8888.pth" \ <br>
--streaming=False \ <br>
--inputPath="./inputTestVideo.mp4" `Or Streaming Url in case of streaming = True` \ <br>
--outputPath="./outputVideo.mp4" \ <br>
--sequenceLength=16 \ <br>
--skip=2 \ <br>
--showInfo=True </p> <br>




<h1 color="green"><b>Instructions to Install our Fight Detection Package</b></h1>
<p>Our Package can be found in this link.
 <a href="https://pypi.org/project/Fight-Detection/0.0.3/">https://pypi.org/project/Fight-Detection/0.0.3/</a></p>

1. Install:

```python

```
2. Download Our Finetuned Model Weights:

```python
import gdown
url = 'https://drive.google.com/uc?id=1MWDeLnpEaZDrKK-OjmzvYLxfjwp-GDcp'
output = 'model_16_m3_0.8888.pth'
gdown.download(url, output, quiet=False)
```

3. Show the Output Video with Detection:

```python

```







<div style="float:left"><img src="https://scontent.fcai20-5.fna.fbcdn.net/v/t39.30808-6/269112292_1642135339476066_5881567363308810890_n.jpg?_nc_cat=110&ccb=1-5&_nc_sid=730e14&_nc_ohc=7NS4qYuWOaoAX8Hln7d&_nc_ht=scontent.fcai20-5.fna&oh=00_AT9eShqku1pSDFMpzapsRWl2X75L5WGtDaO4FvojNyONbA&oe=61C2841F" alt="Your Image"> </div>
