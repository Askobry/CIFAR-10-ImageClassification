import matplotlib.pyplot as plt
import pandas as pd


LeNet_train_acc_path = "./data/run-LeNet-tag-acc_train_acc.csv"
LeNet_val_acc_path = "./data/run-LeNet-tag-acc_val_acc.csv"
LeNet_train_loss_path = "./data/run-LeNet-tag-losses_train_loss.csv"
LeNet_val_loss_path = "./data/run-LeNet-tag-losses_val_loss.csv"
LeNet_lr_path = "./data/run-LeNet-tag-lr.csv"

vgg11_train_acc_path = "./data/run-vgg11-tag-acc_train_acc.csv"
vgg11_val_acc_path = "./data/run-vgg11-tag-acc_val_acc.csv"
vgg11_train_loss_path = "./data/run-vgg11-tag-losses_train_loss.csv"
vgg11_val_loss_path = "./data/run-vgg11-tag-losses_val_loss.csv"
vgg11_lr_path = "./data/run-vgg11-tag-lr.csv"

vgg13_train_acc_path = "./data/run-vgg13-tag-acc_train_acc.csv"
vgg13_val_acc_path = "./data/run-vgg13-tag-acc_val_acc.csv"
vgg13_train_loss_path = "./data/run-vgg13-tag-losses_train_loss.csv"
vgg13_val_loss_path = "./data/run-vgg13-tag-losses_val_loss.csv"
vgg13_lr_path = "./data/run-vgg13-tag-lr.csv"

vgg16_train_acc_path = "./data/run-vgg16-tag-acc_train_acc.csv"
vgg16_val_acc_path = "./data/run-vgg16-tag-acc_val_acc.csv"
vgg16_train_loss_path = "./data/run-vgg16-tag-losses_train_loss.csv"
vgg16_val_loss_path = "./data/run-vgg16-tag-losses_val_loss.csv"
vgg16_lr_path = "./data/run-vgg16-tag-lr.csv"

vgg19_train_acc_path = "./data/run-vgg19-tag-acc_train_acc.csv"
vgg19_val_acc_path = "./data/run-vgg19-tag-acc_val_acc.csv"
vgg19_train_loss_path = "./data/run-vgg19-tag-losses_train_loss.csv"
vgg19_val_loss_path = "./data/run-vgg19-tag-losses_val_loss.csv"
vgg19_lr_path = "./data/run-vgg19-tag-lr.csv"

resnet18_train_acc_path = "./data/run-resnet18-tag-acc_train_acc.csv"
resnet18_val_acc_path = "./data/run-resnet18-tag-acc_val_acc.csv"
resnet18_train_loss_path = "./data/run-resnet18-tag-losses_train_loss.csv"
resnet18_val_loss_path = "./data/run-resnet18-tag-losses_val_loss.csv"
resnet18_lr_path = "./data/run-resnet18-tag-lr.csv"

resnet34_train_acc_path = "./data/run-resnet34-tag-acc_train_acc.csv"
resnet34_val_acc_path = "./data/run-resnet34-tag-acc_val_acc.csv"
resnet34_train_loss_path = "./data/run-resnet34-tag-losses_train_loss.csv"
resnet34_val_loss_path = "./data/run-resnet34-tag-losses_val_loss.csv"
resnet34_lr_path = "./data/run-resnet34-tag-lr.csv"

resnet50_train_acc_path = "./data/run-resnet50-tag-acc_train_acc.csv"
resnet50_val_acc_path = "./data/run-resnet50-tag-acc_val_acc.csv"
resnet50_train_loss_path = "./data/run-resnet50-tag-losses_train_loss.csv"
resnet50_val_loss_path = "./data/run-resnet50-tag-losses_val_loss.csv"
resnet50_lr_path = "./data/run-resnet50-tag-lr.csv"

resnet101_train_acc_path = "./data/run-resnet101-tag-acc_train_acc.csv"
resnet101_val_acc_path = "./data/run-resnet101-tag-acc_val_acc.csv"
resnet101_train_loss_path = "./data/run-resnet101-tag-losses_train_loss.csv"
resnet101_val_loss_path = "./data/run-resnet101-tag-losses_val_loss.csv"
resnet101_lr_path = "./data/run-resnet101-tag-lr.csv"

resnet152_train_acc_path = "./data/run-resnet152-tag-acc_train_acc.csv"
resnet152_val_acc_path = "./data/run-resnet152-tag-acc_val_acc.csv"
resnet152_train_loss_path = "./data/run-resnet152-tag-losses_train_loss.csv"
resnet152_val_loss_path = "./data/run-resnet152-tag-losses_val_loss.csv"
resnet152_lr_path = "./data/run-resnet152-tag-lr.csv"

vit_train_acc_path = "./data/run-vit-tag-acc_train_acc.csv"
vit_val_acc_path = "./data/run-vit-tag-acc_val_acc.csv"
vit_train_loss_path = "./data/run-vit-tag-losses_train_loss.csv"
vit_val_loss_path = "./data/run-vit-tag-losses_val_loss.csv"
vit_lr_path = "./data/run-vit-tag-lr.csv"

cnn_vit_train_acc_path = "./data/run-cnn_vit-tag-acc_train_acc.csv"
cnn_vit_val_acc_path = "./data/run-cnn_vit-tag-acc_val_acc.csv"
cnn_vit_train_loss_path = "./data/run-cnn_vit-tag-losses_train_loss.csv"
cnn_vit_val_loss_path = "./data/run-cnn_vit-tag-losses_val_loss.csv"
cnn_vit_lr_path = "./data/run-cnn_vit-tag-lr.csv"

train_acc_path = [vit_train_acc_path,
                  cnn_vit_train_acc_path]

val_acc_path = [vit_val_acc_path,
                cnn_vit_val_acc_path]

train_loss_path = [vit_train_loss_path,
                   cnn_vit_train_loss_path]

val_loss_path = [vit_val_loss_path,
                 cnn_vit_val_loss_path]

lr_path = [vit_lr_path,
           cnn_vit_lr_path]


colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']

#做平滑处理，我觉得应该可以理解为减弱毛刺，，吧  能够更好地看数据走向
def tensorboard_smoothing(x, smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1,len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x

def plot_acc(train_acc_path, val_acc_path):
    fig, axes = plt.subplots(1, 2, figsize=(10,5))    # a figure with a 1x1 grid of Axes
    
    #设置上方和右方无框
    axes[0].spines['top'].set_visible(False)                   # 不显示图表框的上边框
    axes[0].spines['right'].set_visible(False)  
    axes[1].spines['top'].set_visible(False)                   # 不显示图表框的上边框
    axes[1].spines['right'].set_visible(False)  

    for i, (t, v) in enumerate(zip(train_acc_path, val_acc_path)):
        train = pd.read_csv(t)
        val = pd.read_csv(v)
        axes[0].plot(train['Step'], tensorboard_smoothing(train['Value'], smooth=0.6), color=colors[i], label=t.split('-')[1])
        axes[1].plot(val['Step'], tensorboard_smoothing(val['Value'], smooth=0.6), color=colors[i], label=t.split('-')[1])
        plt.legend(loc = 'lower right')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Train Accuracy")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Val Accuracy")
    # plt.show()


def plot_loss(train_loss_path, val_loss_path):
    fig, axes = plt.subplots(1, 2, figsize=(10,5))    # a figure with a 1x1 grid of Axes
    
    #设置上方和右方无框
    axes[0].spines['top'].set_visible(False)                   # 不显示图表框的上边框
    axes[0].spines['right'].set_visible(False)  
    axes[1].spines['top'].set_visible(False)                   # 不显示图表框的上边框
    axes[1].spines['right'].set_visible(False)  

    for i, (t, v) in enumerate(zip(train_loss_path, val_loss_path)):
        train = pd.read_csv(t)
        val = pd.read_csv(v)
        axes[0].plot(train['Step'], tensorboard_smoothing(train['Value'], smooth=0.6), color=colors[i], label=t.split('-')[1])
        axes[1].plot(val['Step'], tensorboard_smoothing(val['Value'], smooth=0.6), color=colors[i], label=t.split('-')[1])
        plt.legend(loc = 'lower right')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Train Loss")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Val Loss")
    # plt.show()

def plot_lr(lr_path):
    fig, axis = plt.subplots(1, 1, figsize=(5,5))    # a figure with a 1x1 grid of Axes
    
    #设置上方和右方无框
    axis.spines['top'].set_visible(False)                   # 不显示图表框的上边框
    axis.spines['right'].set_visible(False)  

    for i, l in enumerate(lr_path):
        lr = pd.read_csv(l)
        axis.plot(lr['Step'], tensorboard_smoothing(lr['Value'], smooth=0.6), color=colors[i], label=l.split('-')[1])
        plt.legend(loc = 'lower right')
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Learning rate")
        axis.set_title("Learning rate")
    # plt.show()

if __name__ == '__main__':
    plot_acc(train_acc_path, val_acc_path)
    plot_loss(train_loss_path, val_loss_path)
    plot_lr(lr_path)
    plt.show()
