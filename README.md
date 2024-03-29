# Object Extractor

See the objectextractordemo.py for a demo.
See objectextractor.py for the main extractor, all other programs are tools used to aid the synthetic dataset capture + training.

---

From when I was a final year student.

A series of programs used to create synthetic data to artificially inflate existing datasets used in machine learning.
Uses OpenCV and chroma-keying techniques to remove an object to recognise from video output. This is further processed into synthetic
data by blending different permutations of said extracted object with various background images sourced online.

These images are used to train a convolutional neural network to perform a binary classification task; to recognise whether the object
exists within the image or not.

This project was inspired from the need to do image recognition on objects that are unique/do not have many images online, in other-words,
where one cannot source a large dataset. Most hobbyists do not have the luxury of being able to produce large datasets and are limited by
what they can find online, or to painstakingly produce their own with a lot of data capturing. A small paper was written with this program 
the methodology for producing the data/training the model.

To summarise the findings from this experiment, the results indicate that synthetic data can aid in training for image recognition. The 
dataset cannot be completely synthetic, and must be mostly real. But there is a benefit beyond some threshold from having some synthetic
data added. Having complete synthetic data causes the model to learn to spot other synthetic images, likely due to subtle image artifacts.

If given the time to experiment again in the future, I would repeat the experiment with two key differences:
- Produce better synthetic data using the HSV colour model to add different lighting conditions as a new factor in the images
- To use a model already trained to recognise something and test my methodology using transfer learning as opposed to learning from scratch.
