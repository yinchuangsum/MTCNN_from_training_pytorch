import nets
import train

if __name__ == '__main__':
    net = nets.PNet()

    trainer = train.Trainer(net, './param/pnet.pt', r"D:\celeba2\12")
    trainer.train()
