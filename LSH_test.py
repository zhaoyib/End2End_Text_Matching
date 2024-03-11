from Basic_LSH import Basic_LSH
from Multi_Probe_LSH import Multi_Probe_LSH
import torchvision
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dim = 3072
    l = 2
    m = 3
    w = 128

    lsh = Multi_Probe_LSH(dim,l,m,w)

    train_dataset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True)

    for i, data in enumerate(train_dataset.data):
        pic_vec = train_dataset.data[i].ravel()
        lsh.insert(pic_vec, str(i))

    for test_aim in range(100):
        query = test_dataset.data[test_aim].ravel()
        res = lsh.query(query)
        print(res)

        plt.axis('off')
        plt.imshow(test_dataset.data[test_aim])
        plt.savefig('./data/lsh/test_' + str(test_aim) + '.png')

        for i in res:
            plt.imshow(train_dataset.data[int(i)])
            plt.savefig('./data/lsh/test_' + str(test_aim) + '_' + i + '.png')