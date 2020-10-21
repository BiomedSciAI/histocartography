import glob
from matplotlib.patches import Rectangle
from utils import *

class ExtractPatches:
    def __init__(self, config):
        self.config = config
        self.nuclei_types = config.nuclei_types
        self.nuclei_labels = config.nuclei_labels
        self.nuclei_colors = config.nuclei_colors

        idx = self.nuclei_labels.index(-1)
        del self.nuclei_types[-idx]
        del self.nuclei_labels[-idx]
        del self.nuclei_colors[-idx]

        self.patch_size_2 = int(self.config.patch_size/2)
        self.pad = self.patch_size_2
        self.patch_count = np.zeros(len(self.nuclei_types))

        create_directory(self.config.base_patches_path)
        for t in self.nuclei_types[1:]:
            create_directory(self.config.base_patches_path + t.lower())


    def extract_patches(self, is_visualize=True):
        for t in self.config.tumor_types:
            print('Extracting patches from: ', t)

            annotation_centroid_paths = glob.glob(self.config.base_annotation_centroid_path + t + '/*.h5')
            annotation_centroid_paths.sort()

            for i in range(len(annotation_centroid_paths)):
                print(i, '/', len(annotation_centroid_paths))

                basename = os.path.basename(annotation_centroid_paths[i]).split('.')[0]
                image = read_image(self.config.base_img_path + t + '/' + basename + '.png')
                centroids, labels, img_dim = read_centroids(annotation_centroid_paths[i], is_label=True)
                # labels: -1=background, 0=normal, 1=atypical, 2=tumor, 3=stromal, 4=lymphocyte, 5=dead

                image = np.pad(image, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
                                        mode='constant', constant_values=255)

                if is_visualize:
                    fig, ax = plt.subplots(1)
                    ax.imshow(image)

                for j in range(centroids.shape[0]):
                    # Exclude NA nuclei -1)
                    if labels[j] != -1:
                        x = centroids[j, 0] + self.pad
                        y = centroids[j, 1] + self.pad
                        nuclei_type = self.nuclei_types[labels[j]].lower()

                        patch_name = basename + '_' + nuclei_type + '_' + str(centroids[j, 0]) + '_' + str(centroids[j, 1]) + '.png'
                        patch = image[y - self.patch_size_2 : y + self.patch_size_2, x - self.patch_size_2 : x + self.patch_size_2, :]
                        Image.fromarray(patch).save(self.config.base_patches_path + nuclei_type + '/' + patch_name)

                        if is_visualize:
                            rect = Rectangle((x - self.patch_size_2, y - self.patch_size_2),
                                            self.config.patch_size,
                                            self.config.patch_size,
                                            linewidth=2,
                                            edgecolor=self.nuclei_colors[labels[j]],
                                            facecolor='none')
                            ax.add_patch(rect)

                        self.patch_count[labels[j]] += 1

                if is_visualize:
                    plot(image)
                    exit()

        print('\n# Extracted nuclei patches count:')
        for i in range(len(self.nuclei_types)):
            print(self.nuclei_types[i], ':', self.patch_count[i])


