"""Histocartography Pipeline."""
# TODO: put imports here.
import logging
import sys

from .ml.tumor_slide_classification import TumorSlideClassifier
from .io.wsi import load
from .preprocessing.normalization import local_normalization

# setup logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)



# STUB for a tumor classification pipeline that uses other 
# modules of histocartography. In this case, the tumor classification pipeline
# loads and normalizes every wsi from a list, and applies the classifier to it
def tumor_classification_pipeline(input_files=None, classifier=None):
   """Performs the full tumor classification pipeline"""

   log.info(classifier)


   #TODO Predicts based on the loaded files
   for slide in input_files:
      log.info(slide)
      normalized = local_normalization(load(slide)) 
 
      # return classifier(normalized)

   return 'tumor'

