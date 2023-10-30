import numpy as np
from matplotlib import pyplot as plt

showTrainingNumbers = False
training = False

## PART A
## This code was adapted from the provided MATLAB code
## ---------------------------------------------------

trainingData = np.loadtxt("zip.train.txt")

trainingDigits = trainingData[:,0]
trainingGrayscale = trainingData[:,1:]

oneDone = False
fiveDone = False


for i in range(100):
    if trainingDigits[i] == 1 or trainingDigits[i] == 5:

        curimage = np.reshape(trainingGrayscale[i,:], (16, 16))

        if trainingDigits[i] == 1 and not oneDone: 
            curimage1 = curimage
            oneDone = True

        if trainingDigits[i] == 5 and not fiveDone: 
            curimage5 = curimage
            fiveDoneDone = True

if showTrainingNumbers:
    plt.imshow(curimage1)
    plt.figure("Figure 2")
    plt.imshow(curimage5)
    plt.show()

## PART A
## This code was adapted from the provided MATLAB code
## ---------------------------------------------------





## PART B
## ------

## For this we will use symmetry and average intensity to figure out whether we have a 1 or a 5.

## Symmetry will be defined as the difference between the original image and the vertically flipped
## image. AKA, values for grayscale will be vertically flipped and then the difference between this
## new image and the old one will be found via the norm (divided by 256).

## Average intensity will simply be all grayscale values added up and then averaged (once again, divided by 256).


def FindAverageIntensity(image):
    intensity = 0
    for i in range(16):
        for j in range(16):
            intensity = intensity + image[i][j]
    return intensity / 256

def FindSymmetry(image):
    flippedImage = np.flipud(image)
    symmetry = np.linalg.norm(image - flippedImage)
    return symmetry / 256
    
## PART B
## ------





## PART C
## ------


## Aquire training data
trainingImagesOnes = []
trainingImagesFives = []
for i in range(len(trainingData)):

    if trainingDigits[i] == 1:
        curimage = np.reshape(trainingGrayscale[i,:], (16, 16))
        trainingImagesOnes.append(curimage)

    elif trainingDigits[i] == 5:
        curimage = np.reshape(trainingGrayscale[i,:], (16, 16))
        trainingImagesFives.append(curimage)

trainingsymmetryOnes = []
trainingsymmetryFives = []
trainingavgIntensityOnes = []
trainingavgIntensityFives = []


for i in range(len(trainingImagesOnes)):
    trainingsymmetryOnes.append(FindSymmetry(trainingImagesOnes[i]))
    trainingavgIntensityOnes.append(FindAverageIntensity(trainingImagesOnes[i]))

for i in range(len(trainingImagesFives)):
    trainingsymmetryFives.append(FindSymmetry(trainingImagesFives[i]))
    trainingavgIntensityFives.append(FindAverageIntensity(trainingImagesFives[i]))



## Aquire testing data

testingData = np.loadtxt("zip.test.txt")

testingDigits = testingData[:,0]
testingGrayscale = testingData[:,1:]

testingImagesOnes = []
testingImagesFives = []
for i in range(len(testingData)):

    if testingDigits[i] == 1:
        curimage = np.reshape(testingGrayscale[i,:], (16, 16))
        testingImagesOnes.append(curimage)

    elif testingDigits[i] == 5:
        curimage = np.reshape(testingGrayscale[i,:], (16, 16))
        testingImagesFives.append(curimage)

testingsymmetryOnes = []
testingsymmetryFives = []
testingavgIntensityOnes = []
testingavgIntensityFives = []


for i in range(len(testingImagesOnes)):
    testingsymmetryOnes.append(FindSymmetry(testingImagesOnes[i]))
    testingavgIntensityOnes.append(FindAverageIntensity(testingImagesOnes[i]))

for i in range(len(testingImagesFives)):
    testingsymmetryFives.append(FindSymmetry(testingImagesFives[i]))
    testingavgIntensityFives.append(FindAverageIntensity(testingImagesFives[i]))


if training:

    plt.scatter(trainingavgIntensityFives, trainingsymmetryFives, color="red", marker="x", s=30)
    plt.scatter(trainingavgIntensityOnes, trainingsymmetryOnes, color="blue", marker="o", s=30)
    plt.xlabel("Avg Intensity")
    plt.ylabel("Symmetry")
    plt.title("Symmetry over Avg Intensity")
    plt.show()
else:
    plt.scatter(testingavgIntensityFives, testingsymmetryFives, color="red", marker="x", s=30)
    plt.scatter(testingavgIntensityOnes, testingsymmetryOnes, color="blue", marker="o", s=30)
    plt.xlabel("Avg Intensity")
    plt.ylabel("Symmetry")
    plt.title("Symmetry over Avg Intensity")
    plt.show()

## PART C
## ------