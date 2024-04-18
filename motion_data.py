import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt

class ClipDataset(Dataset):
    def __init__(self, root_dir, dtype=torch.float32, device='cpu'):
        self.root_dir = root_dir
        self.dtype=dtype
        self.classes = sorted(os.listdir(root_dir))
        self.classes = [float(self.classes[i]) for i in range(len(self.classes))]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.file_list = self._make_file_list()
        self.device = device

    def _make_file_list(self):
        file_list = []
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith('.p'):
                        file_list.append((os.path.join(class_dir, filename), class_dir))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, class_name = self.file_list[idx]
        # print(file_path)
        frames = torch.as_tensor(pickle.load(open(os.path.join(self.root_dir, file_path), 'rb')), dtype=self.dtype,
                                 device=self.device)
        frames = frames/255

        label = float(class_name)
        return frames, label

def grating(interval):
    row_0 = torch.tensor([0,0,0,0,0,0,0])
    row_1 = torch.tensor([1,0,0,0,0,0,0])
    row_2 = torch.tensor([1,1,0,0,0,0,0])
    row_3 = torch.tensor([1,1,1,0,0,0,0])
    row_4 = torch.tensor([1,1,1,1,0,0,0])
    row_5 = torch.tensor([1,1,1,1,1,0,0])
    row_6 = torch.tensor([1,1,1,1,1,1,0])
    row_7 = torch.tensor([1,1,1,1,1,1,1])
    row_8 = torch.tensor([0,1,1,1,1,1,1])
    row_9 = torch.tensor([0,0,1,1,1,1,1])
    row_10 = torch.tensor([0,0,0,1,1,1,1])
    row_11 = torch.tensor([0,0,0,0,1,1,1])
    row_12 = torch.tensor([0,0,0,0,0,1,1])
    row_13 = torch.tensor([0,0,0,0,0,0,1])

    rows = [row_0, row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8, row_9, row_10, row_11, row_12, row_13]
    num_frames = 14*(interval)
    frames = torch.zeros([num_frames,5,7])
    index = 0
    for i in range(len(rows)):
        frame = torch.vstack([rows[i], rows[i], rows[i], rows[i], rows[i]])
        for _ in range(interval):
            frames[index,:,:] = frame
            index += 1
    return frames

# def grating_set():


if __name__ == '__main__':
    train = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTrain')
    test = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTest')
    sample_img, sample_label = train[0]
    print(train.__len__())
    print(test.__len__())
    print(sample_img.shape, sample_label)

    train_dataloader = DataLoader(train, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=10, shuffle=False)

    imgs, labels = next(iter(train_dataloader))
    print(imgs.shape, labels)
    img = imgs[0,0,:,:]
    print(img.shape)

    interval = 9
    frames = grating(interval)
    print(frames.shape)
    plt.figure()
    for i in range(14):
        plt.subplot(2,7,i+1)
        plt.imshow(frames[i,:,:])
        plt.clim(0,1)

    plt.show()