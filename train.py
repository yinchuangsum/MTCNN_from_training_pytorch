import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sampling import FaceDataset

class Trainer:
    def __init__(self, net, save_path, dataset_path, isCuda=True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda

        if self.isCuda:
            self.net.cuda()


        self.cls_loss_fn = nn.BCELoss() #Binary Loss
        self.offset_loss_fn = nn.MSELoss() #Offset Loss
        self.alli_loss_fn = nn.MSELoss() #alli loss

        self.optimizer = optim.Adam(self.net.parameters(),lr=1e-4)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)


        if os.path.exists(self.save_path): # load weight if exists
            net.load_state_dict(torch.load(self.save_path))


    def train(self):
        faceDataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=50, shuffle=True, num_workers=2)
        # only one epoch
        for _ in range(1):
            for i, (img_data_, category_, offset_, alli_) in enumerate(dataloader):
                if i == 0:
                    print("start training...")
                if self.isCuda:
                    img_data_ = img_data_.cuda()
                    category_ = category_.cuda()
                    offset_ = offset_.cuda()
                    alli_ = alli_.cuda()

                self.scheduler.step()
                _output_category, _output_offset,_output_alli = self.net(img_data_)
                output_category = _output_category.view(-1, 1) # [50,1]
                output_offset = _output_offset.view(-1, 4)     # [50,4]
                output_alli = _output_alli.view(-1,10) #[50,10]


                # calculate loss for label
                category_mask = torch.lt(category_, 2)  #take off part result
                category = torch.masked_select(category_, category_mask)
                output_category = torch.masked_select(output_category, category_mask)
                cls_loss = self.cls_loss_fn(output_category, category)

                # calculate loss for bbox & alli
                offset_mask = torch.gt(category_, 0)   #take off negative result
                offset_index = torch.nonzero(offset_mask)[:, 0]
                alli = alli_[offset_index]
                offset = offset_[offset_index]
                output_offset = output_offset[offset_index]
                output_alli = output_alli[offset_index]
                offset_loss = self.offset_loss_fn(output_offset, offset)
                alli_loss = self.alli_loss_fn(output_alli, alli)


                #total loss
                loss = cls_loss + offset_loss + alli_loss


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #show loss
                print("i=",i ,"loss:", loss.cpu().detach().item(), " cls_loss:", cls_loss.cpu().detach().item(), " offset_loss",
                      offset_loss.cpu().detach().item(), " alli_loss", alli_loss.cpu().detach().item())

        # save
        torch.save(self.net.state_dict(), self.save_path)
        print("save success")





