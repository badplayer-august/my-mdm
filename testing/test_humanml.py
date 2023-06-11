import numpy as np

def main():
    pass

if __name__ == '__main__':
    joint_vec = np.load('dataset/HumanML3D/new_joint_vecs/000000.npy')
    joint = np.load('dataset/HumanML3D/new_joints/000000.npy')
    with open('dataset/HumanML3D/texts/000000.txt', 'r') as text_file:
        text = text_file.read()
    print(joint_vec, joint_vec.shape)
    print(joint, joint.shape)
    print(text)
    print(joint_vec[0][193:-4])
    print(joint[0].reshape(-1))
