import os
import cv2
from PIL import Image
from facenet import FaceNet



if __name__ == '__main__':

    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Colour consts
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2

    datapath = './data'
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Initialize webcam 
    cap = cv2.VideoCapture(0) 

    # initialize a face-net instance
    facenet = FaceNet(anchor_img_path=None,
                      device='cpu',
                      ort=True)
    print("facenet load done\n")

    print('\n============================================')

    # Load images from directory and convert to embeddings for comparison
    for file in os.listdir(datapath + '/match_test'):
        print(file)
        facenet.anchor_img_add(datapath + '/match_test/' + file)

    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale 
        small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        c = 4
        crop = img
        for (x, y, w, h) in faces:
            crop = img[y*c:y*c+h*c, x*c:x*c+w*c]
            min_label, min_ds, min_name = facenet.inference(Image.fromarray(crop))
            cv2.rectangle(img, (x*c, y*c), (x*c+w*c, y*c+h*c), GREEN if min_label[0] == True else RED, 2)
            image = cv2.putText(img, min_name[0].split('.')[0] if min_label[0] else 'Unknown', (x*c, y*c - 10), font, 
                   fontScale, GREEN if min_label[0] == True else RED , thickness, cv2.LINE_AA)
            print(min_label, min_ds, min_name, min_label[0])
            break
        
        # Display
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
        # img = cv2.flip(img, 1)
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    # Release the VideoCapture object
    cap.release()
    
    # Load test image
    # img = Image.open(datapath + '/1251/Dick_Cheney_0001.jpg')
    # label, dist, names = facenet.inference(img)
    # print("onnx inference: label={:} dist={:} names={:}".format(label, dist, names))
    