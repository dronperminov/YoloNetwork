import argparse
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import struct
import cv2
from BoundBox import BoundBox

class Yolo:
	def __init__(self, weights_path):
		self.model = self.make_yolov3_model()
		self.model.load_weights(weights_path)

		# set some parameters
		self.net_h, self.net_w = 416, 416
		self.obj_thresh = 0.5
		self.anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
		self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
				  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
				  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
				  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
				  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
				  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
				  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
				  "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
				  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
				  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


	def _conv_block(self, inp, convs, skip=True):
		x = inp
		count = 0
		
		for conv in convs:
			if count == (len(convs) - 2) and skip:
				skip_connection = x
			count += 1
			
			if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
			x = Conv2D(conv['filter'], 
					   conv['kernel'], 
					   strides=conv['stride'], 
					   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
					   name='conv_' + str(conv['layer_idx']), 
					   use_bias=False if conv['bnorm'] else True)(x)
			if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
			if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

		return add([skip_connection, x]) if skip else x
		
	def _sigmoid(self, x):
		return 1. / (1. + np.exp(-x))

	def make_yolov3_model(self):
		input_image = Input(shape=(None, None, 3))

		# Layer  0 => 4
		x = self._conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
									  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
									  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
									  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

		# Layer  5 => 8
		x = self._conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
							{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
							{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

		# Layer  9 => 11
		x = self._conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
							{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

		# Layer 12 => 15
		x = self._conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
							{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
							{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

		# Layer 16 => 36
		for i in range(7):
			x = self._conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
								{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
			
		skip_36 = x
			
		# Layer 37 => 40
		x = self._conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
							{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

		# Layer 41 => 61
		for i in range(7):
			x = self._conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
								{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
			
		skip_61 = x
			
		# Layer 62 => 65
		x = self._conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
							{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

		# Layer 66 => 74
		for i in range(3):
			x = self._conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
								{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
			
		# Layer 75 => 79
		x = self._conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
							{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
							{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

		# Layer 80 => 82
		yolo_82 = self._conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
								  {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

		# Layer 83 => 86
		x = self._conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
		x = UpSampling2D(2)(x)
		x = concatenate([x, skip_61])

		# Layer 87 => 91
		x = self._conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
							{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
							{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

		# Layer 92 => 94
		yolo_94 = self._conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
								  {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)

		# Layer 95 => 98
		x = self._conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
		x = UpSampling2D(2)(x)
		x = concatenate([x, skip_36])

		# Layer 99 => 106
		yolo_106 = self._conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
								   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
								   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
								   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
								   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
								   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
								   {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

		model = Model(input_image, [yolo_82, yolo_94, yolo_106])    
		return model

	def preprocess_input(self, image):
		new_h, new_w, _ = image.shape

		# determine the new size of the image
		if (float(self.net_w)/new_w) < (float(self.net_h)/new_h):
			new_h = (new_h * self.net_w)/new_w
			new_w = self.net_w
		else:
			new_w = (new_w * self.net_h)/new_h
			new_h = self.net_h

		# resize the image to the new size
		resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

		# embed the image into the standard letter box
		new_image = np.ones((self.net_h, self.net_w, 3)) * 0.5
		new_image[int((self.net_h-new_h)//2):int((self.net_h+new_h)//2), int((self.net_w-new_w)//2):int((self.net_w+new_w)//2), :] = resized
		new_image = np.expand_dims(new_image, 0)

		return new_image

	def decode_netout(self, netout, anchors):
		grid_h, grid_w = netout.shape[:2]
		nb_box = 3
		netout = netout.reshape((grid_h, grid_w, nb_box, -1))
		nb_class = netout.shape[-1] - 5

		boxes = []

		netout[..., :2]  = self._sigmoid(netout[..., :2])
		netout[..., 4:]  = self._sigmoid(netout[..., 4:])
		netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
		netout[..., 5:] *= netout[..., 5:] > self.obj_thresh

		for i in range(grid_h*grid_w):
			row = i / grid_w
			col = i % grid_w
			
			for b in range(nb_box):
				# 4th element is objectness score
				objectness = netout[int(row)][int(col)][b][4]
				#objectness = netout[..., :4]
				
				if(objectness.all() <= self.obj_thresh): continue
				
				# first 4 elements are x, y, w, and h
				x, y, w, h = netout[int(row)][int(col)][b][:4]

				x = (col + x) / grid_w # center position, unit: image width
				y = (row + y) / grid_h # center position, unit: image height
				w = anchors[2 * b + 0] * np.exp(w) / self.net_w # unit: image width
				h = anchors[2 * b + 1] * np.exp(h) / self.net_h # unit: image height  
				
				# last elements are class probabilities
				classes = netout[int(row)][col][b][5:]
				
				box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
				#box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

				boxes.append(box)

		return boxes

	def correct_yolo_boxes(self, boxes, image_h, image_w):
		if (float(self.net_w)/image_w) < (float(self.net_h)/image_h):
			new_w = self.net_w
			new_h = (image_h*self.net_w)/image_w
		else:
			new_h = self.net_w
			new_w = (image_w*self.net_h)/image_h
			
		for i in range(len(boxes)):
			x_offset, x_scale = (self.net_w - new_w)/2./self.net_w, float(new_w)/self.net_w
			y_offset, y_scale = (self.net_h - new_h)/2./self.net_h, float(new_h)/self.net_h
			
			boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
			boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
			boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
			boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

	def _interval_overlap(self, interval_a, interval_b):
		x1, x2 = interval_a
		x3, x4 = interval_b

		if x3 < x1:
			if x4 < x1:
				return 0
			else:
				return min(x2,x4) - x1
		else:
			if x2 < x3:
				 return 0
			else:
				return min(x2,x4) - x3           

	def bbox_iou(self, box1, box2):
		intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
		intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
		
		intersect = intersect_w * intersect_h

		w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
		w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
		
		union = w1*h1 + w2*h2 - intersect
		
		return float(intersect) / union
			
	def do_nms(self, boxes, nms_thresh):
		if len(boxes) > 0:
			nb_class = len(boxes[0].classes)
		else:
			return
			
		for c in range(nb_class):
			sorted_indices = np.argsort([-box.classes[c] for box in boxes])

			for i in range(len(sorted_indices)):
				index_i = sorted_indices[i]

				if boxes[index_i].classes[c] == 0: continue

				for j in range(i+1, len(sorted_indices)):
					index_j = sorted_indices[j]

					if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
						boxes[index_j].classes[c] = 0

	def predict(self, image, nms_thresh = 0.45):
		image_h, image_w, _ = image.shape
		new_image = self.preprocess_input(image)

		yolos = self.model.predict(new_image)
		boxes = []

		for i in range(len(yolos)):
			boxes += self.decode_netout(yolos[i][0], self.anchors[i]) # decode the output of the network

		# correct the sizes of the bounding boxes
		self.correct_yolo_boxes(boxes, image_h, image_w)
		self.do_nms(boxes, nms_thresh) 

		return boxes

	def get_predicted_classes(self, boxes):
		classes = []

		for box in boxes:
			label_str = ''
			label = -1
			
			for i in range(len(self.labels)):
				if box.classes[i] > self.obj_thresh:
					label_str += self.labels[i]
					label = i
					classes.append((self.labels[i], box.classes[i]))

		return classes

	def draw_boxes(self, image, boxes):
		for box in boxes:
			label = -1

			for i in range(len(self.labels)):
				if box.classes[i] > self.obj_thresh:
					label = i
					
			if label >= 0:
				color = (0, 0, 255)

				if box.classes[label] < 0.7:
					color = (0, 0, 255)
				elif box.classes[label] < 0.8:
					color = (0, 255, 255)
				else:
					color = (0, 255, 0)

				cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), color, 3)
				cv2.putText(image, self.labels[label] + ' ' + str(box.get_score()), (box.xmin, box.ymin - 13), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], color, 2)
			
		return image 