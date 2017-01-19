# Nyul Normalizer

The class NyulNormalizer.py implements 

## Dependences
A few dependences are required by NyulNormalier:
* Numpy
* SimpleITK

## Example
Nyul normalization consists in two steps: training and transforming.
During training, the histogram parameters are learned from the training images.

~~~~
from NyulNormalizer import NyulNormalizer

listFiles = ['/path/to/MRI_file1', '/path/to/MRI_file2', ... '/path/to/MRI_fileN']        
outputModel = '/path/to/outputModel'
        
nyul = NyulNormalizer()
        
nyul.train(listFiles)
nyul.saveTrainedModel(outputModel)
~~~~
