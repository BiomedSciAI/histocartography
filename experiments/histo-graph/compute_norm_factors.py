from histocartography.utils.vector import compute_norm
import os

if __name__ == "__main__":

    dir_path = '/Users/frd/Documents/Code/Projects/Experiments/histocartography/data/'

    dirs = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    h5_names = []
    for _dir in dirs:
        g = os.path.join(dir_path, _dir, '_h5')
        files = os.listdir(g)
        for file in files:
            if file.endswith('.h5'):
                h5_names.append(os.path.join(g, file))

    print(len(h5_names))

    norm = compute_norm(h5_names)
    print(norm)

