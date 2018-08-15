# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
import matplotlib.pyplot as plt
import captcha_test
import threading

# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.001

y = []
x = []

plt.figure()
plt.ion()


def draw_line(xv):
    rate = captcha_test.calulate_rate()
    x.append(xv/100)
    y.append(rate)


def main():
    cnn = CNN().cuda()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).cuda()
            labels = Variable(labels.float()).cuda()
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i + 1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
                print("save model")
                try:
                    plt.cla()
                    plt.plot(x, y)
                    plt.pause(0.1)
                except:
                    pass
                threading.Thread(target=draw_line, args=[i]).start()
                # plt.plot(y, x)
                # plt.show()
        # torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
        # print("save model")
        # rate = captcha_test.calulate_rate()
        # x.append(epoch)
        # y.append(rate)
        # plt.plot(x)
        # plt.axis(y)
        # plt.show()
    torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
    print("save last model")


if __name__ == '__main__':
    main()
