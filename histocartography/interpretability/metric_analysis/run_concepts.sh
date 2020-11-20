# Run the unary concepts
ALL_CONCEPTS=('roundness' 'ellipticity' 'crowdedness' 'std_h' 'area')
for concept in "${ALL_CONCEPTS[@]}"
do
    echo "$concept"
    python main.py --explainer -1 --concept $concept --p -1 --rm-misclassification True --nuclei-selection-type thresh --with-nuclei-selection-plot False --rm-non-epithelial-nuclei False --base-path ../../../data/explainability_cvpr/ --distance svm
    sleep 0.1 
done 


# Run the binary concepts
ALL_CONCEPTS=('area,roundness' 'area,ellipticity' 'area,crowdedness' 'crowdedness,area' 'crowdedness,ellipticity' 'roundness,crowdedness')
for concept in "${ALL_CONCEPTS[@]}"
do
    echo "$concept"
    python main.py --explainer -1 --concept $concept --p -1 --rm-misclassification True --nuclei-selection-type thresh --with-nuclei-selection-plot False --rm-non-epithelial-nuclei False --base-path ../../../data/explainability_cvpr/ --distance svm
    sleep 0.1 
done 


# Run on all concepts
ALL_CONCEPTS=('roundness,ellipticity,crowdedness,std_h,area')
for concept in "${ALL_CONCEPTS[@]}"
do
    echo "$concept"
    python main.py --explainer -1 --concept $concept --p -1 --rm-misclassification True --nuclei-selection-type thresh --with-nuclei-selection-plot False --rm-non-epithelial-nuclei False --base-path ../../../data/explainability_cvpr/ --distance svm
    sleep 0.1 
done 