from srgan.srgan_utils import SRGAN_predict,save_images_test 
import argparse 
import time 
from keras.models import load_model 
 
 
 
parser = argparse.ArgumentParser(description='Image Super Resolution') 
parser.add_argument('--modelPath') 
parser.add_argument('--inputPath') 
parser.add_argument('--outputPath') 
 
 
 
 
def main(): 
    # parsing args 
    args = parser.parse_args() 
    # model= keras.models.load_model(args.modelPath)   
    start=time.time() 
    low_image,generated_Image = SRGAN_predict(args.inputPath,args.modelPath) 
    save_images_test(low_image,generated_Image,args.outputPath) 
    end = time.time() 
     
    print("Total time of predictaion is : ",end-start) 
    print(generated_Image) 
    return generated_Image 
 
 
if name == '__main__': 
 main() 
 
 
#For Inference
#python -m infer --modelPath="generator_5000.h5" --inputPath="5.png" --outputPath="SRImage55.png"