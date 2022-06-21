from torchvision import transforms

transform = transforms.Compose([
    transforms.CenterCrop(100),
    transforms.ToTensor()
])

train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="/dccstor/vww_dataset/dataset/all2014", 
                    annFile="/dccstor/vww_dataset/dataset/visualwakewords-dataset/annotations/instances_train.json", transform= transform) 
valid_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="/dccstor/vww_dataset/dataset/all2014", annFile="/dccstor/vww_dataset/dataset/visualwakewords-dataset/annotations/instances_val.json", transform= transform)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=1)
