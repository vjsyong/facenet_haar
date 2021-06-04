import os
import sys
import cv2
from time import time

from PIL import Image


from facenet import FaceNet

if __name__ == '__main__':

    datapath = './data'    

    test_file0 = []
    test_file1 = []
    test_result = []
    test_dist = []
    test_ct = []
    test_result_ort = []
    test_dist_ort = []
    test_ct_ort = []    

    # # cpu turn
    # # initialize a face-net instance
    # facenet = FaceNet(anchor_img_path=None,
    #                   device='cpu',
    #                   ort=False)
    # print("facenet load done\n")

    # print('\n============================================')
    
    # for item in os.listdir(datapath):
    #     path = datapath + '/' + item
    #     if os.path.isdir(path):
    #         # clear anchor embedding in db
    #         facenet.clear_db()
    #         k = 0
    #         for file in os.listdir(path):
    #             file = path + '/' + file
    #             if os.path.isfile(file):
                    
    #                 print('%d %s' % (k, file))

    #                 if k == 0:
    #                     test_file0.append(file)
    #                     st = time()
    #                     print("anchor_img_add: start")
    #                     facenet.anchor_img_add(file)
    #                     ct = time() - st
    #                     print("anchor_img_add: ct={:}\n".format(ct))
    #                 else:
    #                     test_file1.append(file)
    #                     img = Image.open(file)
    #                     st = time()
    #                     print("onnx inference: start")
    #                     label, dist, names = facenet.inference(img)
    #                     ct = time() - st
    #                     print("onnx inference: label={:} dist={:} names={:} ct={:}\n".format(label, dist, names, ct))

    #                     test_result.append(label[0])
    #                     test_dist.append(dist[0])
    #                     test_ct.append(ct)
                    
    #                 k += 1

    # ort turn
    # initialize a face-net instance
    facenet = FaceNet(anchor_img_path=None,
                      device='cpu',
                      ort=True)
    print("facenet load done\n")

    print('\n============================================')
    
    for item in os.listdir(datapath):
        path = datapath + '/' + item
        print(path)
        if os.path.isdir(path):
            # clear anchor embedding in db
            facenet.clear_db()
            k = 0
            for file in os.listdir(path):
                file = path + '/' + file
                if os.path.isfile(file):
                    
                    print('%d %s' % (k, file))

                    if k == 0:
                        #test_file0.append(file)
                        st = time()
                        print("anchor_img_add: start")
                        facenet.anchor_img_add(file)
                        ct = time() - st
                        print("anchor_img_add: ct={:}\n".format(ct))
                    else:
                        #test_file1.append(file)
                        img = Image.open(file)
                        st = time()
                        print("onnx inference: start")
                        label, dist, names = facenet.inference(img)
                        ct = time() - st
                        print("onnx inference: label={:} dist={:} names={:} ct={:}\n".format(label, dist, names, ct))

                        test_result_ort.append(label[0])
                        test_dist_ort.append(dist[0])
                        test_ct_ort.append(ct)
                    
                    k += 1

    # test summary
    print("\n-----------------------------------------------------")
    print("Test Results:")
    for file0, file1, result, result_ort, dist, dist_ort, ct, ct_ort in zip(test_file0, test_file1, test_result, test_result_ort, test_dist, test_dist_ort, test_ct, test_ct_ort):
        print("%s %s" % (file0, file1))
        print("cpu:\t{:}\t{:}\t{:}s".format(result, dist, ct))
        print("ort:\t{:}\t{:}\t{:}s".format(result_ort, dist_ort, ct_ort))
        print("\n")