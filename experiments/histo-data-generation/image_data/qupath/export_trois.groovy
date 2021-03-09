/**
 * Script to export image regions corresponding to all annotations of an image.
 * Note: Pay attention to the 'downsample' value to control the export resolution!
*/

import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Define downsample value for export resolution
def downsample = 1.0
def imageExportType = 'PNG'

// Output directory
base_path = '/Users/pus/Desktop/pascale/trois/'
mkdirs(base_path)

def benign_path = base_path + '0_benign'
def pathological_benign_path = base_path + '1_pathological_benign'
def udh_path = base_path + '2_udh'
def adh_path = base_path + '3_adh'
def fea_path = base_path + '4_fea'
def dcis_path = base_path + '5_dcis'
def malignant_path = base_path + '6_malignant'

mkdirs(benign_path)
mkdirs(pathological_benign_path)
mkdirs(udh_path)
mkdirs(adh_path)
mkdirs(fea_path)
mkdirs(dcis_path)
mkdirs(malignant_path)

// Get the main QuPath data structures
def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// Class names. Be consistent with the annotation names.
benignClass = getPathClass("Benign-sure")
pathologicalbenignClass = getPathClass("Pathological-benign-sure")
udhClass = getPathClass("UDH-sure")
adhClass = getPathClass("ADH-sure")
feaClass = getPathClass("FEA-sure")
dcisClass = getPathClass("DCIS-sure")
malignantClass = getPathClass("Malignant-sure")

// Print annotation information
infoAnnotations()

// Export each annotation
print('Extracting troi images...')
for (annotation in getAnnotationObjects()){
    pathClass = annotation.getPathClass()

    if (pathClass == benignClass)
        pathOutput = benign_path
    else if (pathClass == pathologicalbenignClass)
        pathOutput =  pathological_benign_path
    else if (pathClass == udhClass)
        pathOutput =  udh_path
    else if (pathClass == adhClass)
        pathOutput =  adh_path
    else if (pathClass == feaClass)
        pathOutput =  fea_path
    else if (pathClass == dcisClass)
        pathOutput =  dcis_path
    else if (pathClass == malignantClass)
        pathOutput =  malignant_path

    saveImage(pathOutput, server, annotation, downsample, imageExportType)
}
print 'Done!'

/**
 * Save extracted image region corresponding to an object ROI.
 *
 * @param pathOutput Directory in which to store the output
 * @param server ImageServer for the relevant image
 * @param pathObject The object to export
 * @param downsample Downsample value for the export of both image region & mask
 * @param imageExportType Type of image (original pixels, not mask!) to export ('JPG', 'PNG' or null)
 * @return
 */
def saveImage(String pathOutput, ImageServer server, PathObject pathObject, double downsample, String imageExportType) {
    // Extract ROI & classification name
    def roi = pathObject.getROI()
    def pathClass = pathObject.getPathClass()
    def classificationName = pathClass == null ? 'None' : pathClass.toString()
    if (roi == null) {
        print 'Warning! No ROI for object ' + pathObject + ' - cannot export corresponding region'
        return
    }

    // Create a region from the ROI
    def region = RegionRequest.createInstance(server.getPath(), downsample, roi)

    // Create a name
    String name = String.format('%s_%s_%.0f_%d_%d_%d_%d',
            server.getShortServerName(),
            classificationName,
            region.getDownsample(),
            region.getX(),
            region.getY(),
            region.getWidth(),
            region.getHeight()
    )

    // Request the BufferedImage
    def img = server.readBufferedImage(region)

    // Create filename & export
    if (imageExportType != null) {
        def fileImage = new File(pathOutput, name + '.' + imageExportType.toLowerCase())
        ImageIO.write(img, imageExportType, fileImage)
    }
}

def infoAnnotations() {
    // Count the number of annotations
    annotations = getAnnotationObjects()
    print("Total number of annotations: " + annotations.size())
    
    nBenign = 0
    nPathologicalBenign = 0
    nUDH = 0
    nADH = 0
    nFEA = 0
    nDCIS = 0
    nMalignant = 0
    
    for (annotation in getAnnotationObjects()) {
        pathClass = annotation.getPathClass()
        if (pathClass == benignClass)
          nBenign++
        else if (pathClass == pathologicalbenignClass)
          nPathologicalBenign++
        else if (pathClass == udhClass)
          nUDH++
        else if (pathClass == adhClass)
          nADH++
        else if (pathClass == feaClass)
          nFEA++
        else if (pathClass == dcisClass)
          nDCIS++
        else if (pathClass == malignantClass)
          nMalignant++
        else
          print("Error: " + pathClass)
    }
    print("BENIGN: " + nBenign)
    print("PATHOLOGICAL BENIGN: " + nPathologicalBenign)
    print("UDH: " + nUDH)
    print("ADH: " + nADH)
    print("FEA: " + nFEA)
    print("DCIS: " + nDCIS)
    print("MALIGNANT: " + nMalignant)
}
