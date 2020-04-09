from architectures.DANet.model.MSDualGuided import *
from torch.utils.data import DataLoader
from datasets.ChaosDataset import *
from torchvision import transforms
from torch.optim import Adam


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    chaos_dataset = ChaosDataset(mode="train", root_dir="../rawdata/CHAOS_", transform=transform, augment=Augment())
    train_loader = DataLoader(chaos_dataset, batch_size=8, num_workers=0, shuffle=True)

    net = MSDualGuided()
    net.apply(weights_init)

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    #Dice_loss = computeDiceOneHot()
    mseLoss = nn.MSELoss()

    if torch.cuda.is_available():
        net.cuda()
        softMax.cuda()
        CE_loss.cuda()
        #Dice_loss.cuda()

    optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99), amsgrad=False)

    for i in range(10):
        net.train()
        for (image, mask) in train_loader:
            image, mask = image.cuda(), mask.cuda()

            semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2 = net(image)
            segmentation_prediction = sum(list(predict1) + list(predict2)) / 8
            predClass_y = softMax(segmentation_prediction)

            optimizer.zero_grad()

            loss = loss_gt(target, output_gt)

            loss.backward()
            optimizer.step()

