# Nyul Normalizer

The class NyulNormalizer.py implements 

## Dependences
A few dependences are required by NyulNormalier:
* Numpy
* SimpleITK

## Example
Nyul normalization consists in two steps: training and transforming.
During training, the histogram parameters are learned from the training images.

```python
from NyulNormalizer import NyulNormalizer

listFiles = ['/path/to/MRI_file1.nii.gz', '/path/to/MRI_file2.nii.gz', ... '/path/to/MRI_fileN.nii.gz']
outputModel = '/path/to/outputModel'
        
nyul = NyulNormalizer()
        
nyul.train(listFiles)
nyul.saveTrainedModel(outputModel)
```

Once a model is trained, we can load it and transform any image:

```python
from NyulNormalizer import NyulNormalizer

nyul = NyulNormalizer()
nyul.loadTrainedModel('/path/to/saved/model.npz')
nyul.transformImage('/path/to/image.nii.gz')
```