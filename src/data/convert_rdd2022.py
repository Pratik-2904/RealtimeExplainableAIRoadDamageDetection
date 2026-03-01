import os
import glob
import xml.etree.ElementTree as ET
import argparse

# RDD2022 primary classes
CLASSES = ["D00", "D10", "D20", "D40"]

def convert_voc_to_yolo(xml_path, output_txt_path):
    """
    Parses Pascal VOC XML structure and converts bounding boxes to YOLO format.
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        if size is None:
            return
            
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        if w == 0 or h == 0:
            return
            
        with open(output_txt_path, 'w') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    continue
                    
                cls = obj.find('name').text
                if cls not in CLASSES:
                    continue
                    
                cls_id = CLASSES.index(cls)
                xmlbox = obj.find('bndbox')
                
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)
                
                # Convert to normalized YOLO format
                x_center = ((xmin + xmax) / 2.0) / w
                y_center = ((ymin + ymax) / 2.0) / h
                b_width = (xmax - xmin) / w
                b_height = (ymax - ymin) / h
                
                # Clamp coordinates just in case of annotation errors
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                b_width = max(0.0, min(1.0, b_width))
                b_height = max(0.0, min(1.0, b_height))
                
                out_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {b_width:.6f} {b_height:.6f}\n")
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RDD2022 Pascal VOC XMLs to YOLO TXT format")
    parser.add_argument("--xml_dir", type=str, help="Directory containing downloaded XML files")
    parser.add_argument("--txt_dir", type=str, help="Output directory for generated TXT files")
    args = parser.parse_args()
    
    if args.xml_dir and args.txt_dir:
        os.makedirs(args.txt_dir, exist_ok=True)
        xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))
        
        print(f"Found {len(xml_files)} XML files. Starting conversion to YOLO format...")
        for xml_file in xml_files:
            filename = os.path.basename(xml_file)
            txt_filename = filename.replace('.xml', '.txt')
            txt_path = os.path.join(args.txt_dir, txt_filename)
            convert_voc_to_yolo(xml_file, txt_path)
        print("Conversion complete!")
    else:
        print("Usage: python convert_rdd2022.py --xml_dir <path> --txt_dir <path>")
