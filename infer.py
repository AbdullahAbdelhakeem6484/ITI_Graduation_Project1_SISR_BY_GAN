from srgan.srgan_utils import SRGAN_predict
import argparse
import time
from keras.models import load_model



parser = argparse.ArgumentParser(description='Image Super Resolution')
parser.add_argument('--modelPath')
parser.add_argument('--inputPath')
# parser.add_argument('--outputPath')




def main():
	# parsing args
	args = parser.parse_args()
	# model= keras.models.load_model(args.modelPath)  

	start=time.time()
	generated_Image = SRGAN_predict(args.inputPath,args.modelPath)
	end = time.time()
	print(end-start)
	return generated_Image


if __name__ == '__main__':
	main()


#pip install keras==2.3.1
#pip install tensorflow==2.1.0
#python -m infer --modelPath="generator_5000.h5" --inputPath="5.png"