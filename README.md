# Nyul Normalizer

The class NyulNormalizer.py implements a method for intensity normalization 
as explained in [1] and [2].

## Dependences
A few dependences are required by NyulNormalier:
* Numpy (`pip install numpy`)
* Scipy (`pip install scipy`)
* Matplotlib (`pip install matplotlib`)
* SimpleITK (`pip install SimpleITK`)

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

## References

**[1]** Nyúl, László G., and Jayaram K. Udupa. ["On standardizing the MR image intensity scale"](https://www.ncbi.nlm.nih.gov/pubmed/10571928) Magn Reson Med. (1999) Dec;42(6):1072-81.

**[2]** Shah, Mohak, et al. ["Evaluating intensity normalization on MRIs of human brain with multiple sclerosis"](http://www.sciencedirect.com/science/article/pii/S1361841510001337) Medical image analysis 15.2 (2011): 267-282.
