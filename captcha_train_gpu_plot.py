# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
import captcha_test
import time
import threading

# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.001

y = []
x = []


def draw():
    while True:
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.figure(1)
            plt.cla()
            plt.plot(x, y)
            plt.pause(1)
        finally:
            pass


thread = threading.Thread(target=draw)
thread.start()



def main():
    cnn = CNN().cuda()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        time_time = time.time()
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).cuda()
            labels = Variable(labels.float()).cuda()
            predict_labels = cnn(images)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            calculate_rate(i)
            # if (i + 1) % 10 == 0:
            #     print("epoch:", epoch, "step:", i, "loss:", loss.item())
            # if (i + 1) % 100 == 0:
            #     torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
            #     print("save model")
        print("epoch %d %.2fs" % (epoch, time.time() - time_time))
        torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
        print("save model")
        calculate_rate(epoch)
    torch.save(cnn.state_dict(), "./model.pkl")  # current is model.pkl
    print("save last model")


def calculate_rate(epoch):
    rate = captcha_test.calulate_rate()
    x.append(epoch)
    y.append(rate)


if __name__ == '__main__':
    main()
