from PIL import Image, ImageOps, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

savePath = os.getcwd() + os.sep + "data_preprocess"

os.chdir("data")

def resizeImage(imagePath, folder):
    im = Image.open(os.getcwd() + os.sep +imagePath).convert('RGB')
    im = ImageOps.fit(im, (200, 200), Image.ANTIALIAS, (0.5, 0.5))
    #im = ImageOps.grayscale(im)
    im.save(savePath + os.sep + folder + os.sep + imagePath)

initialPath = os.getcwd()
for folder in os.listdir(initialPath):
    if folder.startswith("."):
        continue
    os.chdir(folder)
    try:
        os.makedirs(savePath + os.sep + folder)
    except:
        pass    
    for img in os.listdir(os.getcwd()):
        if img.startswith("."):
            continue
        if os.path.isfile(os.getcwd() + os.sep + img):
            print img
            resizeImage(img, folder)
    os.chdir(initialPath)
