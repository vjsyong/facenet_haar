import os
import time

import torch
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from torch.nn.modules.distance import PairwiseDistance

from scipy.io import savemat, readsav
from models.inception_resnet_v1 import InceptionResnetV1
from PIL import Image
import numpy as np

import cv2
import onnx
import onnxruntime

class FaceNet:
    def __init__(self, device='cuda', anchor_img_path=None, ort=True):
        device=torch.device(device)
        self.device = device
        self.ort = ort
        if not self.ort:
            self.model, self.best_distance_threshold = self.load_model()
            self.model.eval()
            self.model.to(device)
        else:
            # tune for accuracy
            # best_distance_threshold = 0.8  # 200515:fix false +ve for Chi as LIS (faceid_demo_trt_nano1_200515)
            self.best_distance_threshold = 0.75
            self.onnx_name = 'facenet.onnx'
            # load onnx 
            self.onnx_model = onnx.load(self.onnx_name)
            onnx.checker.check_model(self.onnx_model)
            print("%s check done\n" % self.onnx_name)
            # setup onnx runtime inference session
            self.ort_session = onnxruntime.InferenceSession(self.onnx_name)
            print("%s session setup done\n" % self.onnx_name)

        self.input_size = 160
        self.preprocessing1 = transforms.Compose([
            transforms.Resize(size=[self.input_size, self.input_size])
        ])
        self.distance = PairwiseDistance(2).to(device)		
        self.anchor_img = []
        self.anc_embedding = []
        self.anchor_img_name_list = []
        if anchor_img_path is not None:
            load_db(anchor_img_path)
        print('len(self.anc_embedding)={:}'.format(len(self.anc_embedding)))

    def export_onnx(self):
        # onnx conversion codes
        dummy = torch.randn(1, 3, self.input_size, self.input_size)
        torch.onnx.export(self.model, dummy, self.onnx_name)
        print("%s created\n" % self.onnx_name)

    def load_db(self, anchor_img_path='data/'):
        self.anchor_img_name_list = self.read_image_db(anchor_img_path)
        print('len(self.anchor_img_name_list)={:}'.format(len(self.anchor_img_name_list)))
        self.anchor_img = []
        self.anc_embedding = []
        for k in range(len(self.anchor_img_name_list)):
            print('img_name[{:}]={:}'.format(k, self.anchor_img_name_list[k]))  
            _anchor_img = Image.open(self.anchor_img_name_list[k])
            _anchor_img = self.preprocessing1(_anchor_img)
            _anchor_img = F.to_tensor(np.float32(_anchor_img))
            _anchor_img = torch.unsqueeze((_anchor_img - 127.5) / 128.0, 0)
            if not self.ort:
                _anchor_embedding = self.model(_anchor_img.to(self.device))
            else:
                ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(_anchor_img)}
                ort_outs = self.ort_session.run(None, ort_inputs)
                output1 = ort_outs[0]
                _anchor_embedding = torch.from_numpy(output1)
            self.anchor_img.append(_anchor_img)
            self.anc_embedding.append(_anchor_embedding)
        print('len(self.anc_embedding)={:}'.format(len(self.anc_embedding)))
        
    def clear_db(self):
        self.anchor_img = []
        self.anc_embedding = []
        self.anchor_img_name_list = []

    @staticmethod
    def load_model():
        # tune for accuracy
        # best_distance_threshold = 0.8  # 200515:fix false +ve for Chi as LIS (faceid_demo_trt_nano1_200515)
        best_distance_threshold = 0.75
        model = InceptionResnetV1().eval()
        name = 'vggface2'
        if name == 'casia-webface':
            model_name = "20180408-102900-casia-webface.pt"
        else:
            model_name = "20180402-114759-vggface2.pt"
        state_dict = torch.load(os.path.join('./checkpoints/', model_name))
        model.load_state_dict(state_dict)

        return model, best_distance_threshold

    @staticmethod
    def toTensor(img):
        return F.to_tensor(np.float32(img) - 127.5) / 128.0

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
    @staticmethod
    def read_image_db(path):
        if not os.path.isdir(path):
            print("folder does not exist!")
            return None
        image_names = os.listdir(path)
        print("Total images in DB: {:}".format(len(image_names)))
        image_name_list = []
        for i in range(0, len(image_names)):
            if image_names[i].endswith("png") or image_names[i].endswith("jpg"):
                image_name_list.append(os.path.join(path, image_names[i]))
        return image_name_list

    def anchor_img_add(self, anchor_img_name):
        print('anchor_img_name={:}'.format(anchor_img_name))
        self.anchor_img_name_list.append(anchor_img_name)        
        _anchor_img = Image.open(anchor_img_name)
        _anchor_img = self.preprocessing1(_anchor_img)
        _anchor_img = F.to_tensor(np.float32(_anchor_img))
        _anchor_img = torch.unsqueeze((_anchor_img - 127.5) / 128.0, 0)
        if not self.ort:
            _anchor_embedding = self.model(_anchor_img.to(self.device))
        else:
            ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(_anchor_img)}
            ort_outs = self.ort_session.run(None, ort_inputs)
            output1 = ort_outs[0]
            _anchor_embedding = torch.from_numpy(output1)
        self.anchor_img.append(_anchor_img)
        self.anc_embedding.append(_anchor_embedding)

    def inference(self, test_img_pil):
        with torch.no_grad():
            test_img = self.preprocessing1(test_img_pil)
            test_img = F.to_tensor(np.float32(test_img))
            test_img = torch.unsqueeze((test_img - 127.5) / 128.0, 0)
            if not self.ort:
                test_embedding = self.model(test_img.to(self.device))
            else:
                ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(test_img)}
                ort_outs = self.ort_session.run(None, ort_inputs)
                output1 = ort_outs[0]
                test_embedding = [torch.from_numpy(output1)]

            min_ds = []
            min_name = []
            min_label = []
            if len(self.anc_embedding) == 0:
                dummy = torch.zeros([1,512], requires_grad=False)
                ds = self.distance.forward(dummy, test_embedding[0]).cpu().detach().numpy()
                min_ds.append(ds)
                min_name.append('None')
                min_label.append(False)
            else:
                for i in range(len(test_embedding)):
                    min_ds.append(100)
                    min_name.append("")
                    min_label.append(False)
                    min_k = -1
                    for k in range(len(self.anc_embedding)):
                        ds = self.distance.forward(self.anc_embedding[k], test_embedding[i]).cpu().detach().numpy()
                        if ds < min_ds[i]:
                            min_ds[i] = ds
                            min_k = k
                    if min_k != -1:
                        name = self.anchor_img_name_list[min_k]
                        base = os.path.basename(name)
                        min_name[i] = os.path.splitext(base)[0]
                        if min_ds[i] < self.best_distance_threshold:
                            min_label[i] = True
            return min_label, min_ds, min_name

    def inference2(self, test_img, tag_id):
        with torch.no_grad():
            test_img = self.preprocessing1(test_img_pil)
            test_img = F.to_tensor(np.float32(test_img))
            test_img = torch.unsqueeze((test_img - 127.5) / 128.0, 0)
            if not self.ort:
                test_embedding = self.model(test_img.to(self.device))
            else:
                ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(test_img)}
                ort_outs = self.ort_session.run(None, ort_inputs)
                output1 = ort_outs[0]
                test_embedding = [torch.from_numpy(output1)]

            min_ds = []
            min_name = []
            min_label = []
            for i in range(len(test_embedding)):
                min_ds.append(100)
                min_name.append("")
                min_label.append(False)
                min_k = -1
                for k in range(len(self.anc_embedding)):
                    name = self.anchor_img_name_list[k]
                    base = os.path.basename(name)
                    name = os.path.splitext(base)[0]
                    #print("len tag_id = %d" % len(tag_id))
                    #print("tag_id {:}".format(tag_id))
                    #print("name={:}".format(name))
                    if len(tag_id) > 0:
                        # with tag ids specified, process records with tagid info only
                        s = name.split('_')[0][0:3]
                        if s.lower() == 'tag':
                            id = int(name.split('_')[0][3:]) # remove 'tag' at the beginning
                            print("id=%d" % id)
                            name2 = name[len(name.split('_')[0])+1:] # remove tagXXX_                            
                            for tid in tag_id:
                                if tid == id:
                                    ds = self.distance.forward(self.anc_embedding[k], test_embedding[i]).cpu().detach().numpy()
                                    if ds < min_ds[i]:
                                        min_ds[i] = ds
                                        min_name[i] = name2
                                        min_k = k
                    else:
                        # no tag ids specified
                        s = name.split('_')[0][0:3]
                        if s.lower() == 'tag':
                            name2 = name[len(name.split('_')[0])+1:] # remove tagXXX_
                        else:
                            name2 = name
                            
                        print("name2=%s" % name2)
                        
                        ds = self.distance.forward(self.anc_embedding[k], test_embedding[i]).cpu().detach().numpy()
                        if ds < min_ds[i]:
                            min_ds[i] = ds
                            min_name[i] = name2
                            min_k = k

                if min_k != -1:
                    if min_ds[i] < self.best_distance_threshold:
                        min_label[i] = True

            print("min_name={:}".format(min_name))
            return min_label, min_ds, min_name

    def inference_numpy(self, anchor_img, test_img):
        anc_embedding = self.model(anchor_img)
        test_embedding = self.model(test_img)
        # use numpy operation
        ds = (anc_embedding.cpu().numpy() - test_embedding.cpu().numpy())
