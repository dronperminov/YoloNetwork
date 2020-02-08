import argparse
import cv2

from Yolo import Yolo

argparser = argparse.ArgumentParser(description='yolo network')
argparser.add_argument('-i', '--image', help='path to image file')
    
if __name__ == '__main__':
    args = argparser.parse_args()

    yolo = Yolo("weights.h5")

    image = cv2.imread(args.image) # preprocess the image
    boxes = yolo.predict(image) # run the prediction

    objects = yolo.get_predicted_classes(boxes)

    for label, probability in objects:
        print(label, ':', probability)
        
    yolo.draw_boxes(image, boxes) # draw bounding boxes on the image using labels
 
    cv2.imwrite(args.image[:-4] + '_predicted' + args.image[-4:], (image).astype('uint8')) # write the image with bounding boxes to file