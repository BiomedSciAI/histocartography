import glob
import time
import copy
from matplotlib.patches import Circle
from utils import *


class ExtractAnnotation:
    def __init__(self, config):
        self.config = config
        self.nuclei_types = self.config.nuclei_types
        self.nuclei_colors = self.config.nuclei_colors
        self.tumor_types = self.config.tumor_types

        # Input
        self.base_img_path = self.config.base_img_path
        self.base_masks_path = self.config.base_masks_path
        self.base_centroid_path = self.config.base_centroid_path
        self.base_instance_map_path = self.config.base_instance_map_path

        # Output
        self.base_annotation_centroid_path = self.config.base_annotation_centroid_path
        self.base_overlaid_path = self.config.base_overlaid_path


    def get_nuclei_annotations(self, is_visualize=False):
        nuclei_count = np.zeros(len(self.nuclei_types))
        total_time = time.time()

        for t in self.tumor_types:
            print('****************************************************************** Tumor type: ', t)
            create_directory(self.base_annotation_centroid_path + t)
            create_directory(self.base_overlaid_path + t)

            annotation_masks_path = glob.glob(self.base_masks_path + t + '/*.png')
            annotation_masks_path.sort()

            for f in annotation_masks_path:
                start_time = time.time()
                basename = os.path.basename(f).split('.')[0]
                centroid_path = self.base_centroid_path + t + '/' + basename + '.h5'
                instance_map_path = self.base_instance_map_path + t + '/_h5/' + basename + '.h5'

                # Input
                annotation_mask = np.array(Image.open(f))
                if os.path.isfile(centroid_path) and os.path.isfile(instance_map_path):
                    centroids, img_dim = read_centroids(centroid_path)
                    instance_map = read_instance_map(instance_map_path)
                else:
                    print('File doesnt exist: ', basename)
                    continue

                labels = self.get_labels(centroids, instance_map, annotation_mask)
                # labels: -1=NA, 0=normal, 1=atypical, 2=tumor, 3=stromal, 4=lymphocyte, 5=dead

                # Counting
                (unique, count) = np.unique(labels, return_counts=True)
                unique = unique.astype(int)
                for i in range(len(unique)):
                    nuclei_count[unique[i] + 1] += count[i]

                # Saving
                save_info(self.base_annotation_centroid_path + t + '/' + basename + '.h5',
                          keys=['instance_centroid_location', 'instance_centroid_label', 'image_dimension'],
                          values=[centroids, labels, img_dim],
                          dtypes=['float32', 'int32', 'int32'])

                if is_visualize:
                    self.visualize(basename)

                print(basename, ': time=', round(time.time() - start_time, 2))

        print('\n# Annotated nuclei count:')
        nuclei_count = nuclei_count.astype(int)
        for i in range(len(self.nuclei_types)):
            print(self.nuclei_types[i], ':', nuclei_count[i])

        print('Total time: ', round(time.time() - total_time, 2))


    def get_labels(self, centroids, instance_map, annotation_mask):
        # Get unique nuclei instances
        (unique_nuclei, count_nuclei) = np.unique(instance_map, return_counts=True)
        unique_nuclei = unique_nuclei[1:]
        count_nuclei = count_nuclei[1:]

        # Get annotated nuclei instances
        # annotation_mask: 0=background, 1=normal, 2=atypical, 3=tumor, 4=stromal, 5=lymphocyte, 6=dead
        mask = copy.deepcopy(annotation_mask)
        mask[mask > 0] = 1      # Remove background
        mask = instance_map * mask
        (unique_ann, count_ann) = np.unique(mask, return_counts=True)
        unique_ann = unique_ann[1:]
        count_ann = count_ann[1:]

        # Get label per nuclei instance
        labels = np.ones(centroids.shape[0]) * -1

        for i in range(len(unique_nuclei)):
            if unique_nuclei[i] in unique_ann:
                # Check overlap of annotated nuclei with actual nuclei size
                ratio = count_ann[unique_ann == unique_nuclei[i]]/ count_nuclei[i]
                if ratio >= 0.9:
                    (x, y) = centroids[i]
                    labels[i] = annotation_mask[y, x] - 1
                    # Subtract 1 to have labels: -1=NA, 0=normal, 1=atypical, 2=tumor, 3=stromal, 4=lymphocyte, 5=dead
        return labels


    def visualize(self, basename):
        tumor_type = basename.split('_')[1]
        img = read_image(self.base_img_path + tumor_type + '/' + basename + '.png')

        centroids, labels, img_dim = read_centroids(self.base_annotation_centroid_path + tumor_type + '/' + basename + '.h5', is_label=True)
        # labels: -1=NA, 0=normal, 1=atypical, 2=tumor, 3=stromal, 4=lymphocyte, 5=dead

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(img)

        for i in range(centroids.shape[0]):
            if labels[i] != -1:
                x = centroids[i, 0]
                y = centroids[i, 1]
                circ = Circle((x, y), 5, fill=False, color=self.nuclei_colors[labels[i] + 1], linewidth=2)
                ax.add_patch(circ)

        plt.axis('off')
        plt.savefig(self.base_overlaid_path + tumor_type + '/' + basename + '.png', dpi=600, bbox_inches='tight')
        plt.close()




























