import tqdm
import torch
import torch.optim as optim
from model import YOLOv3
from config import *
from loss import YOLOLoss
from dataset import Dataset, test_transform
from utils import load_checkpoint, convert_cells_to_bboxes, nms, plot_image

# Taking a sample image and testing the model

# Setting the load_model to True
load_model = True

# Defining the model, optimizer, loss function and scaler
model = YOLOv3().to(device)
optimizer = optim.Adam(model.parameters(), lr=leanring_rate)
loss_fn = YOLOLoss()
scaler = torch.cuda.amp.GradScaler()

# Loading the checkpoint
if load_model:
    load_checkpoint(checkpoint_file, model, optimizer, leanring_rate)

# Defining the test dataset and data loader
test_dataset = Dataset(
    csv_file="./data/pascal voc/test.csv",
    image_dir="./data/pascal voc/images/",
    label_dir="./data/pascal voc/labels/",
    anchors=ANCHORS,
    transform=test_transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=2,
    shuffle=True,
)

# Getting a sample image from the test data loader
x, y = next(iter(test_loader))
x = x.to(device)

model.eval()
with torch.no_grad():
    # Getting the model predictions
    output = model(x)
    # Getting the bounding boxes from the predictions
    bboxes = [[] for _ in range(x.shape[0])]
    anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    # Getting bounding boxes for each scale
    for i in range(3):
        batch_size, A, S, _, _ = output[i].shape
        anchor = anchors[i]
        boxes_scale_i = convert_cells_to_bboxes(
            output[i], anchor, s=S, is_predictions=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
model.train()

# Plotting the image with bounding boxes for each image in the batch
for i in range(batch_size):
    # Applying non-max suppression to remove overlapping bounding boxes
    nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6)
    # Plotting the image with bounding boxes
    plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)
