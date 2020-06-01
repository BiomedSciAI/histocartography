from histocartography.utils.vector import compute_norm
import os

if __name__ == "__main__":

    dir_path = '/dataT/pus/histocartography/Data/PASCALE_NEW/nuclei_info/nuclei_vae_features'

    dirs = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    h5_names = []
    for _dir in dirs:
        print(_dir)
        g = os.path.join(dir_path, _dir)
        files = os.listdir(g)
        for file in files:
            if file.endswith('.h5'):
                h5_names.append(os.path.join(g, file))

    print(len(h5_names))

    norm = compute_norm(h5_names)
    print(norm)
