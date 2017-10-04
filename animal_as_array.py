import os, numpy
from scipy.misc import imread

ANIMAL_PROCESSED_PATH = os.getcwd() + os.sep + "data_preprocess"

def get_animals():

    # Read animal names, paths.
    animal_names, animal_paths = [], []
    for animal_name in sorted(os.listdir(ANIMAL_PROCESSED_PATH)):
        folder = ANIMAL_PROCESSED_PATH + os.sep + animal_name
        if not os.path.isdir(folder):
            continue
        paths = [os.path.join(folder, f) for f in os.listdir(folder) if f != '.DS_Store']
        n_images = len(paths)
        animal_names.extend([animal_name] * n_images)
        animal_paths.extend(paths)

    n_animal = len(animal_paths)
    target_names = numpy.unique(animal_names)
    target = numpy.searchsorted(target_names, animal_names)

    # Read image data.
    animals = numpy.zeros((n_animal, 100, 100, 3), dtype=numpy.float32)
    for i, animal_path in enumerate(animal_paths):
        img = imread(animal_path)
        animal = numpy.asarray(img, dtype=numpy.uint32)
        animals[i, ...] = animal

    indices = numpy.arange(n_animal)
    numpy.random.RandomState(42).shuffle(indices)
    animals, target = animals[indices], target[indices]

    return {
        'data': animals.reshape(len(animals), -1),
        'images': animals,
        'target': target,
        'target_names': target_names,
        #'DESCR': 'Animals'
    }

retrieved_data = get_animals()
#print len(retrieved_data['data'])
#print len(retrieved_data['images'])
