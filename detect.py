import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import utils
import nets
from torchvision import transforms
import time
import os
import math

#image > image pyramid > Pnet > nms > Rnet > nms > Onet >nms >draw

# confidence and nms threshold
# Pnet:
p_cls = 0.6
p_nms = 0.5
# Rnet：
r_cls = 0.6
r_nms = 0.5
# Onet：
o_cls = 0.95
o_nms = 0.5


class Detector():

    def __init__(self, pnet_param="./param/pnet.pt", rnet_param="./param/rnet.pt", onet_param="./param/onet.pt",
                 isCuda=False):

        self.isCuda = isCuda

        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param, map_location= "cpu" ))
        self.rnet.load_state_dict(torch.load(rnet_param, map_location= "cpu" ))
        self.onet.load_state_dict(torch.load(onet_param, map_location= "cpu" ))

        self.__image_transform = transforms.Compose([transforms.ToTensor()]) #NCHW and normalized

    def detect(self,image):

        #run pnet
        start_time = time.time() #test time
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0: #no face detected
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time

        #run rnet
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        #run onet
        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:  # no value return
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        # check time usage
        t_sum = t_pnet + t_rnet + t_onet
        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __pnet_detect(self, img): # any image size can enter fully convolution
        total_boxes = np.array([]) # empty boxes
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1 # initial scale
        while min_side_len > 12: #stop at 12pixel
            img_data = self.__image_transform(img) #img to tensor
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0) # add C dimension

            _cls, _offest,_ = self.pnet(img_data)

            cls = _cls[0][0].cpu().data
            offset = _offest[0].cpu().data
            idxs = torch.gt(cls, p_cls) # compare with confidence threshold
            idx = torch.nonzero(idxs,as_tuple=False)
            boxes = self.__box(idx, offset[:, idxs], cls[idxs], scale)

            boxes = utils.nms(np.array(boxes), p_nms) #perform iou
            scale *= 0.7 # resize
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)
            if boxes.shape[0] != 0:
                total_boxes = np.vstack([total_boxes,boxes]) if total_boxes.size else boxes

        return total_boxes

    # convert back to original image
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12): #effective stride = all strides multiply

        _x1 = (start_index[:, 1].float() * stride) / scale
        _y1 = (start_index[:, 0].float() * stride) / scale
        _x2 = (start_index[:, 1].float() * stride + side_len) / scale
        _y2 = (start_index[:, 0].float() * stride + side_len) / scale

        ow = _x2 - _x1  # actual box width and height
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[0] #bounding box location
        y1 = _y1 + oh * offset[1]
        x2 = _x2 + ow * offset[2]
        y2 = _y2 + oh * offset[3]

        boxes = torch.stack((x1, y1, x2, y2, cls)).permute(1,0)
        return boxes  # bounding box location and confidence

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes) # convert box to square
        for _box in _pnet_boxes: # for loop is slow
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2)) # crop
            img = img.resize((24, 24)) # resize
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset, _ = self.rnet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        idxs, _ = np.where(cls > r_cls)
        _box = _pnet_boxes[idxs]
        _x1 = _box[:,0].astype('int32')
        _y1 = _box[:,1].astype('int32')
        _x2 = _box[:,2].astype('int32')
        _y2 = _box[:,3].astype('int32')

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[idxs,0]
        y1 = _y1 + oh * offset[idxs,1]
        x2 = _x2 + ow * offset[idxs,2]
        y2 = _y2 + oh * offset[idxs,3]

        boxes = np.stack((x1,y1,x2,y2,cls[idxs].T[0])).T

        return utils.nms(np.array(boxes), r_nms)


    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset, _alli = self.onet(img_dataset)
        cls = _cls.cpu().data.numpy()       # (1, 1)
        offset = _offset.cpu().data.numpy() # (1, 4)
        alli = _alli.cpu().data.numpy()

        idxs, _ = np.where(cls > o_cls)
        _box = _rnet_boxes[idxs]
        _x1 = _box[:, 0].astype('int32')
        _y1 = _box[:, 1].astype('int32')
        _x2 = _box[:, 2].astype('int32')
        _y2 = _box[:, 3].astype('int32')

        ow = _x2 - _x1
        oh = _y2 - _y1

        x1 = _x1 + ow * offset[idxs, 0]
        y1 = _y1 + oh * offset[idxs, 1]
        x2 = _x2 + ow * offset[idxs, 2]
        y2 = _y2 + oh * offset[idxs, 3]
        le_x = _x1 + ow * alli[idxs, 0]
        le_y = _y1 + oh * alli[idxs, 1]
        re_x = _x2 + ow * alli[idxs, 2]
        re_y = _y1 + oh * alli[idxs, 3]
        n_x = _x1 + ow * alli[idxs, 4]
        n_y = _y1 + oh * alli[idxs, 5]
        lm_x = _x1 + ow * alli[idxs, 6]
        lm_y = _y2 + oh * alli[idxs, 7]
        rm_x = _x2 + ow * alli[idxs, 8]
        rm_y = _y2 + oh * alli[idxs, 9]

        boxes = np.stack((x1, y1, x2, y2, cls[idxs].T[0], le_x, le_y, re_x, re_y, n_x, n_y, lm_x, lm_y, rm_x, rm_y)).T

        return utils.nms(np.array(boxes), o_nms, isMin=True) #iou is divided by smallest area no union

def rotate(image):
    w,h = image.size
    if w > h :
        return image.rotate(-90,expand = True)

if __name__ == '__main__':
    image_path = r"test_images"
    output_path = r"output"
    for i in os.listdir(image_path):
        detector = Detector()
        with Image.open(os.path.join(image_path,i)) as im:
            print(i)
            print("----------------------------")
            im.load()
            im = rotate(im)
            boxes = detector.detect(im)
            boxes = utils.convert_to_square(boxes)
            print("size:",im.size)
            imDraw = ImageDraw.Draw(im)
            cx = im.size[0]/2
            cy = im.size[1]/2
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                le_x = int(box[5])
                le_y = int(box[6])
                re_x = int(box[7])
                re_y = int(box[8])
                n_x = int(box[9])
                n_y = int(box[10])
                lm_x = int(box[11])
                lm_y = int(box[12])
                rm_x = int(box[13])
                rm_y = int(box[14])
                print((x1, y1, x2, y2))
                print("conf:",box[4])
                #draw bounding boxes and landmark
                imDraw.rectangle((x1, y1, x2, y2), outline='red')
                imDraw.ellipse((utils.draw_circle(le_x, le_y)), fill='red')
                imDraw.ellipse((utils.draw_circle(re_x, re_y)), fill='red')
                imDraw.ellipse((utils.draw_circle(n_x, n_y)), fill='blue')
                imDraw.ellipse((utils.draw_circle(lm_x, lm_y)), fill='green')
                imDraw.ellipse((utils.draw_circle(rm_x, rm_y)), fill='green')

            im.save(os.path.join(output_path,i))



