import cv2
import xml.etree.ElementTree as ET
import os

w = 1280
h = 720

def voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymin = int(float(bbox.find('ymin').text))
        ymax = int(float(bbox.find('ymax').text))
        boxes.append((name,(xmin,ymin,xmax,ymax)))
    return boxes

def yolo_txt(txt_file):
    boxes = []
    with open(txt_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            # print(data)

            # center_x = float(data[1])*w
            # center_y = float(data[2])*h

            xmin = int((float(data[1])-float(data[3])/2)*w)
            xmax = int((float(data[1])+float(data[3])/2)*w)
            ymin = int((float(data[2])-float(data[4])/2)*h)
            ymax = int((float(data[2])+float(data[4])/2)*h)
            name = data[0]
            boxes.append((name,(xmin,ymin,xmax,ymax)))
    return boxes

def draw_boxes(image_path, path):
    image = cv2.imread(image_path)
    # boxes = voc_xml(path)
    boxes = yolo_txt(path)

    for name, (xmin,ymin,xmax,ymax) in boxes:
        if(name == "0"):
            cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(0,255,0),5)
        elif(name=="1"):
            cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(0,0,255),5)

    cv2.imshow('bbox', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_frame(image_path, path):
    image = cv2.imread(image_path)
    # boxes = voc_xml(path)
    boxes = yolo_txt(path)

    for name, (xmin,ymin,xmax,ymax) in boxes:
        if(name == "0"):
            cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(0,255,0),1)
        elif(name=="1"):
            cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(0,0,255),1)
    return image

def rvt_txt(txt_file):
    boxes = []
    with open(txt_file, "r", encoding="utf-8") as file:
        file.readline()
        lines = file.readlines()
        for line in lines:
            data = line.strip().split(",")
            # print(data)

            # center_x = float(data[1])*w
            # center_y = float(data[2])*h

            xmin = int(float(data[1]))
            xmax = int((float(data[1])+float(data[3])))
            ymin = int(float(data[2]))
            ymax = int((float(data[2])+float(data[4])))
            frame = int(data[0])
            name = data[5]
            boxes.append((name,(xmin,ymin,xmax,ymax), frame))
    return boxes