// SOURCE:
// Exporting annotations
// https://qupath.readthedocs.io/en/latest/docs/advanced/exporting_annotations.html

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

def name = imageData.getServer().getFile().getName().split("\\.")[0]
def tumor_type = name.split("_")[1]
def width = server.getWidth()
def height =  server.getHeight()

def pathOutput = "/Users/pus/Desktop/Projects/Data/Histocartography/Nuclei/annotation_masks/" + tumor_type + "/" + name + ".png"
double downsample = 1.0

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .downsample(downsample)                 // Choose server resolution
    .backgroundLabel(0, ColorTools.BLACK)   // Specify background label (usually 0 or 255)
    .addLabel('Normal', 1)                  // Choose output labels (the order matters!)
    .addLabel('Atypical', 2)
    .addLabel('Tumor', 3)
    .addLabel('Stromal', 4)
    .addLabel('Lymphocyte', 5)
    .addLabel('Dead', 6)
    .build()

// Write the image
writeImage(labelServer, pathOutput)

print('DONE')















