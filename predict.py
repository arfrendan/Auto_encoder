import numpy as np
import keras
from PIL import Image
import matplotlib.pyplot as plt
from model import SegNet
from keras.utils import plot_model
import dataset

height = 360
width = 480
classes = 12
epochs = 100
batch_size = 1
log_filepath='./logs_100/'

data_shape = 360*480
class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
def writeImage(image, filename):
    """ label data to colored image """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def predict(test):
    model = keras.models.load_model('seg.h5')
    model.summary()
    probs = model.predict(test, batch_size=1)
    for pro in probs:
        prob = pro.reshape((height,width,classes)).argmax(axis = 2)
        plt.matshow(prob)
        plt.show()
    prob = probs[0].reshape((height, width, classes)).argmax(axis=2)
    return prob

def estimate(testX,testY):
    ds = dataset.Dataset(classes=classes)
    train_X, train_y = ds.load_data('train')
    train_X = ds.preprocess_inputs(train_X)
    train_Y = ds.reshape_labels(train_y)
    model = keras.models.load_model('seg.h5')
    model.fit(train_X, train_Y, batch_size=1, epochs=3,
              verbose=1, class_weight=class_weighting, validation_data=(testX, testY), shuffle=True)
def main():
    print("loading data...")
    ds = dataset.Dataset(test_file='val.txt', classes=classes)
    test_X, test_y = ds.load_data('test') # need to implement, y shape is (None, 360, 480, classes)
    test_X = ds.preprocess_inputs(test_X)
    test_Y = ds.reshape_labels(test_y)
    plt.matshow(test_Y[0].reshape((height,width,classes)).argmax(axis = 2))
    plt.show()
    # prob = predict(test_X)
    # writeImage(prob, 'val.png')

    # estimate(test_X, test_Y)
if __name__ == '__main__':
    main()