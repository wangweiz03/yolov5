import json
import os

def json_to_yaml(json_file):   
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract the list of images and labels
    labels = data['categories']

    for idx, ins in enumerate(labels):
        print(f"  {idx}: {ins['name']}")
            
json_to_yaml('E:/Courses/数字图像处理/hw3/yolov5/custom_dataset/annotations.json')

def json_to_txt(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建图像ID到文件名的映射
    id_to_image = {img['id']: img for img in data['images']}
    
    print(id_to_image)
    # 创建类别ID到序号的映射（YOLO需要从0开始的连续编号）
    cat_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(data['categories'])}
    # print(cat_id_to_yolo_id)
    
    # 按图像分组标注
    image_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann) # 'annotations'是一个列表，包含该图像的所有标注
    
    # 为每个图像创建TXT文件
    # print(image_annotations)
    for img_id, anns in image_annotations.items():
        img_info = id_to_image[img_id]
        txt_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        txt_path = output_dir + '/' + txt_filename
        
        with open(txt_path, 'w') as f:
            for ann in anns: # 'annotations'是一个列表，包含该图像的所有标注
                # 原始bbox格式 [x, y, width, height] (绝对像素值)
                x, y, w, h = ann['bbox']
                
                # 转换为YOLO格式 [x_center, y_center, width, height] (归一化)
                img_w, img_h = img_info['width'], img_info['height']
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # 获取YOLO格式的类别ID
                yolo_class_id = cat_id_to_yolo_id[ann['category_id']]
                
                # 写入文件
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# 使用示例
json_path = '../custom_dataset/annotations.json'
output_dir = '../custom_dataset/labels'  # 输出目录，应与images目录同级
json_to_txt(json_path, output_dir)
    