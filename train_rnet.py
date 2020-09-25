import nets
import train

if __name__ == '__main__':
    net = nets.RNet()

    trainer = train.Trainer(net, './param/rnet.pt', r"D:\celeba2\24")
    trainer.train()
