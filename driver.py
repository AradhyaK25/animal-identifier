import video_process
import classifier
import os
import numpy
from PIL import Image, ImageOps, ImageFile
from scipy.misc import imread

video_path = "concat.mp4"

video_process.record_frames(video_path)

working_dir = os.getcwd()
os.chdir("video_results/%s" % video_path)
os.makedirs(working_dir + os.sep + "video_temp" + os.sep + video_path)
for img in os.listdir(os.getcwd()):
    if img.startswith("."):
        continue
    if os.path.isfile(os.getcwd() + os.sep + img):
        im = Image.open(os.getcwd() + os.sep +img).convert('RGB')
        im = ImageOps.fit(im, (300, 300), Image.ANTIALIAS, (0.5, 0.5))
        im.save(working_dir + os.sep + "video_temp" + os.sep + video_path + os.sep + img)

# Read image data.
PROCESSED_PATH = working_dir + os.sep + "video_temp" + os.sep + video_path
paths = [os.path.join(PROCESSED_PATH, f) for f in os.listdir(PROCESSED_PATH) if f != '.DS_Store']
n_animal = len(paths)
animals = numpy.zeros((n_animal, 300, 300,3), dtype=numpy.float32)
for i, animal_path in enumerate(paths):
    img = imread(animal_path)
    animal = numpy.asarray(img, dtype=numpy.uint32)
    animals[i, ...] = animal

animals = animals.reshape(len(animals), -1)

X_test = animals

os.chdir(working_dir)
predicted_animals = classifier.predict(X_test)
print predicted_animals
