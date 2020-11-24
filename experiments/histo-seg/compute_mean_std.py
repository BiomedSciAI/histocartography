import pandas as pd

from eth import *
from utils import *
from tqdm.auto import tqdm

processed_image_path = (PREPROCESS_PATH / "VahadaneStainNormalizer(lambda_c=0.01,lambda_s=0.1,target=ZT111_4_C_7_1,thres=0.8)")
df = merge_metadata(pd.read_pickle(IMAGES_DF), pd.read_pickle(ANNOTATIONS_DF), add_image_sizes=True, processed_image_directory=processed_image_path)
df = df[df.slide.isin([111, 199, 204])]

downsample_factor = 3
image_paths = df["processed_image_path"].tolist()
images = list()
for image_path in tqdm(
    image_paths,
    total=len(image_paths),
    desc="dataset_loading",
):
    image = read_image(image_path)
    if downsample_factor != 1:
        new_size = (
            image.shape[0] // downsample_factor,
            image.shape[1] // downsample_factor,
        )
        image = cv2.resize(image, new_size)
    images.append(image)
images = torch.from_numpy(np.array(images))
images = images.to(torch.float32) / 255

mean = images.mean(axis=(1,2)).mean(axis=0)
print(*map(lambda x: x.item(), mean))
std = images.std(axis=(1,2)).std(axis=0)
print(*map(lambda x: x.item(), std))
torch.save(mean, 'img_mean.pth')
torch.save(std, 'img_std.pth')
