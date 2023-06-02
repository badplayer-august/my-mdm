from data_loaders.get_data import get_dataset_loader
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    data = get_dataset_loader(batch_size=64, num_frames=60)
    for motion, cond in tqdm(data):
        print(motion)
        print(motion.shape)
        print(cond)
        joint_vec = np.load('dataset/HumanML3D/new_joint_vecs/000000.npy')
        break
