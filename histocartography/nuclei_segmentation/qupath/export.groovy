// Read csv file
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.roi.RectangleROI
import qupath.lib.scripting.QPEx

def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()
def name = server.getShortServerName()

base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/nuclei_segmentation/Predictions/_csv/'
def path = base_path + name + '.csv'
def fh = new File(path).getText('UTF-8')
def lines = fh.split('\n')
N = lines.size()
print(N)

w = 6 as Double
h = 6 as Double

for (i=0; i<N; i++){
    temp = lines[i].split(',')
    x = temp[0] as Double
    y = temp[1] as Double
    
    x = x - w/2
    y = y - h/2
    
    // Create a new Rectangle ROI
    def roi = new RectangleROI(x, y, w, h)
    
    // Create & new annotation & add it to the object hierarchy
    def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass('None'))    
    imageData.getHierarchy().addPathObject(annotation, true)
}





