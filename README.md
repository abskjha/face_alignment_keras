# face_alignment_keras
This is the keras implementation of the face alignment network. Build using [FAN](https://www.adrianbulat.com)'s state-of-the-art deep learning based face alignment method. 

* This repository also contains training code of training your own model.
* This network has been modified for fine grained lip-landmark localization, to adap it for full face landmark localization, change utils.py and model.py files.

**Note:** 

* The lua version, by the authors is available [here](https://github.com/1adrianb/2D-and-3D-face-alignment).

* The pytorch version, by the authors is available [here](https://github.com/1adrianb/face-alignment).


#### Get the Face Alignment source code
```bash
git clone https://github.com/abskjha/face_alignment_keras
```

### Requirements

* Python 3.5+ or Python 2.7 (it may work with other versions too)
* Linux, Windows or macOS
* Keras (2.1.3)
* Tensorflow (1.8.0)


## Contributions

All contributions are welcomed. If you encounter any issue (including examples of images where it fails) feel free to open an issue.

## Citation

```
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
```

For citing dlib, pytorch or any other packages used here please check the original page of their respective authors.