import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling2D

from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import TSNE

import os
import glob
import pickle
from pathlib import Path
home = str(Path.home())

mnet = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
mnet.summary()

model = GlobalMaxPooling2D()(mnet.layers[-1].output)
model = Model(inputs=mnet.get_input_at(0), outputs=model)

paths = glob.glob(home + '/datasets/mscoco/val2014/*.jpg')
paths = list(paths)
paths.sort()
paths = paths[:2000]

fig = plt.figure(1)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

feats = []
pklpath = '/tmp/feats.pkl'

if not os.path.exists(pklpath):
    for idx, path in enumerate(paths):
        print(idx, path)
        img = cv.imread(path)
        img = cv.resize(img, (224, 224))
        img = np.expand_dims(img, 0)
        img = preprocess_input(img)
        feat = model.predict(img)
        feats.append(feat.squeeze())
    pickle.dump(feats, open(pklpath, 'wb'))
else:
    feats = pickle.load(open(pklpath, 'rb'))


def on_click(event):
    x, y = event.xdata, event.ydata
    print(x, y)
    min_dist = 100000
    for idx, feat2d in enumerate(feats2d):
        dist =  np.linalg.norm(feat2d-[x, y])
        if dist < min_dist:
            min_dist = dist
            min_path = paths[idx]

    print(min_path)
    img = cv.imread(min_path)
    img = cv.resize(img, (224, 224))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ax2.imshow(img)
    plt.draw()


feats2d = PCA(n_components=2).fit_transform(feats)
#feats2d = TSNE(n_components=2).fit_transform(feats)

ax1.scatter(feats2d[:, 0], feats2d[:, 1])
cid = ax1.figure.canvas.mpl_connect('button_press_event', on_click)

plt.show()




