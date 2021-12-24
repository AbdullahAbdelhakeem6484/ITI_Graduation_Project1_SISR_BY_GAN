from setuptools import setup, find_packages

setup(name='SRGAN_ITI',
version='0.0.1',
author='Abdullah Abdelhakeem',
packages=find_packages(),
install_requires=[
    "PIL",
    "imageio",
    "keras",
    "matplotlib",
    "numpy",
    "pandas",
    "session_info",
    "skimage",
    "tensorflow",
    "IPython",
    "Python",

],
zip_safe=False,
)