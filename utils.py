import os
import shutil
import cv2
import xml.etree.ElementTree as ET
import random
import shutil

from glob import glob
from config import *
from PIL import ImageOps, Image


def join_path(base_path, relative_path):
    return os.path.join(base_path, relative_path)

def get_sample_list_from_path(path, sample_format):
    if os.path.exists(path):
        return glob(join_path(path, f'*{sample_format}'))
    
def format_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def make_dirs_or_format_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        format_folder(folder_path)

def make_dirs_for_table_extraction(folder_path):
    TD_FOLDER = join_path(folder_path, 'TD')
    make_dirs_or_format_dir(TD_FOLDER)
    make_dirs_or_format_dir(join_path(TD_FOLDER, 'images'))
    make_dirs_or_format_dir(join_path(TD_FOLDER, 'train'))
    make_dirs_or_format_dir(join_path(TD_FOLDER, 'val'))
    make_dirs_or_format_dir(join_path(TD_FOLDER, 'test'))

    TSR_FOLDER = join_path(folder_path, 'TSR')
    make_dirs_or_format_dir(TSR_FOLDER)
    make_dirs_or_format_dir(join_path(TSR_FOLDER, 'images'))
    make_dirs_or_format_dir(join_path(TSR_FOLDER, 'train'))
    make_dirs_or_format_dir(join_path(TSR_FOLDER, 'val'))
    make_dirs_or_format_dir(join_path(TSR_FOLDER, 'test'))

    return TD_FOLDER, TSR_FOLDER

def random_differrence_ratio(value_range):
    return random.uniform(value_range[0], value_range[1])

def move_file(base_path, task_dest_path, task_name, file):
    BASE_PATH = join_path(base_path, file + '.xml')
    DEST_PATH = join_path(join_path(task_dest_path, task_name), file + '.xml')
    shutil.move(BASE_PATH, DEST_PATH)

def split_file(data_split, data_split_name, td_dir, tsr_dir):
    data_index = -1
    for data in data_split:
        data_index +=1
        for element in data:
            move_file(td_dir[0], td_dir[1], data_split_name[data_index], element)
            move_file(tsr_dir[0], tsr_dir[1], data_split_name[data_index], element)

def get_tree_and_root_from_file_xml(path):
    tree = ET.parse(path)
    return tree, tree.getroot()

def get_class_list_from_root(root, class_name):
    return root.findall(f'.//object[name="{class_name}"]') 

def get_bounding_box_object(object):
    return {e.tag : int(e.text) for e in object.find('bndbox')}

def scale_dimension(input_width, input_height, fixed_dimension_length):
    if input_width > input_height:
        std_width = fixed_dimension_length
        std_height = round(input_height * (fixed_dimension_length / input_width))
    elif input_width < input_height:
        std_width = round(input_width * (fixed_dimension_length / input_height))
        std_height = fixed_dimension_length
    else:
        std_width = fixed_dimension_length
        std_height = fixed_dimension_length
    
    return std_width, std_height

def normalize_td_image_size(img_dir, label_dir, output_td_dir):
    base_name = os.path.basename(img_dir)
    sample_name = os.path.splitext(base_name)[0]
    image_name = sample_name
    
    # std_width = 0
    # std_height = 0

    org_img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    img_height, img_width = org_img.shape[:2]

    std_width, std_height = scale_dimension(img_width, img_height, MAX_LENGTH_PER_DIMENSION)
    
    std_img = cv2.resize(org_img, (std_width, std_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(join_path(output_td_dir, image_name + '.jpg'), std_img)

    # Get original tree & root from file .xml
    org_tree, org_root = get_tree_and_root_from_file_xml(label_dir)

    # Get table list
    org_tabel_list = get_class_list_from_root(org_root, CLASS_TABLE)

    # AUTOMATIC IMAGE LABELING
    td_root = ET.Element("annotation")
    ET.SubElement(td_root, "filename").text = image_name + '.jpg'
    ET.SubElement(td_root, "folder").text = 'TD PROJECT HOON'
    ET.SubElement(td_root, "path").text = join_path(output_td_dir, image_name + '.jpg')
    source = ET.SubElement(td_root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(td_root, "size")
    ET.SubElement(size, "width").text = str(MAX_LENGTH_PER_DIMENSION)
    ET.SubElement(size, "height").text = str(MAX_LENGTH_PER_DIMENSION)
    ET.SubElement(size, "depth").text = "3"

    for table in org_tabel_list:
        xmin_table, ymin_table, xmax_table, ymax_table = get_bounding_box_object(table).values()
        
        x_ratio = std_width / img_width
        y_ratio = std_height / img_height

        new_xmin = round(x_ratio * xmin_table)
        new_ymin = round(y_ratio * ymin_table)
        new_xmax = round(x_ratio * xmax_table)
        new_ymax = round(y_ratio * ymax_table)

        # Label class table
        object_table = ET.SubElement(td_root, "object")
        ET.SubElement(object_table, "name").text = "table"
        ET.SubElement(object_table, "pose").text = "Unspecified"
        ET.SubElement(object_table, "truncated").text = "0"
        ET.SubElement(object_table, "difficult").text = "0"
        bndbox_table = ET.SubElement(object_table, "bndbox")
        ET.SubElement(bndbox_table, "xmin").text = str(new_xmin)
        ET.SubElement(bndbox_table, "ymin").text = str(new_ymin)
        ET.SubElement(bndbox_table, "xmax").text = str(new_xmax)
        ET.SubElement(bndbox_table, "ymax").text = str(new_ymax)
    
    td_tree = ET.ElementTree(td_root)
    xml_dir = join_path(output_td_dir, image_name + '.xml')
    td_tree.write(xml_dir)

def automatic_recalculate_bbox_class_for_tsr(root, class_object, class_name, ratio, padding):
    x_ratio = ratio[0]
    y_ratio = ratio[1]

    original_padding_x, original_padding_y = padding[0]
    new_padding_x, new_padding_y = padding[1]

    org_xmin_class, org_ymin_class, org_xmax_class, org_ymax_class = get_bounding_box_object(class_object).values()

    new_xmin_class = round(x_ratio * (org_xmin_class - original_padding_x) + new_padding_x)
    new_ymin_class = round(y_ratio * (org_ymin_class - original_padding_y) + new_padding_y)
    new_xmax_class = round(x_ratio * (org_xmax_class - original_padding_x) + new_padding_x)
    new_ymax_class = round(y_ratio * (org_ymax_class - original_padding_y) + new_padding_y)

    class_object = ET.SubElement(root, "object")
    ET.SubElement(class_object, "name").text = class_name
    ET.SubElement(class_object, "pose").text = "Unspecified"
    ET.SubElement(class_object, "truncated").text = "0"
    ET.SubElement(class_object, "difficult").text = "0"
    bndbox_class = ET.SubElement(class_object, "bndbox")
    ET.SubElement(bndbox_class, "xmin").text = str(new_xmin_class)
    ET.SubElement(bndbox_class, "ymin").text = str(new_ymin_class)
    ET.SubElement(bndbox_class, "xmax").text = str(new_xmax_class)
    ET.SubElement(bndbox_class, "ymax").text = str(new_ymax_class)    

def normalize_tsr_image_size(img_dir, label_dir, output_tsr_dir):
    base_name = os.path.basename(img_dir)
    sample_name = os.path.splitext(base_name)[0]
    image_name = sample_name

    # Get original tree & root from file .xml
    org_tree, org_root = get_tree_and_root_from_file_xml(label_dir)

    org_img_width = float(org_root.find('size/width').text)
    org_img_height = float(org_root.find('size/height').text)

    # Get table list
    org_tabel_list = get_class_list_from_root(org_root, CLASS_TABLE)

    for table in org_tabel_list:
        # Get bounding box of class table
        org_xmin_table, org_ymin_table, org_xmax_table, org_ymax_table = get_bounding_box_object(table).values()
        ORIGINAL_PADDING_X = org_xmin_table
        ORIGINAL_PADDING_Y = org_ymin_table

        # Get original image
        org_img = Image.open(img_dir)

        # Calculate dimensions of resize image and resize
        resized_img_width, resized_img_height = scale_dimension(org_img_width, org_img_height, MAX_LENGTH_PER_DIMENSION)
        resized_img = org_img.resize((resized_img_width, resized_img_height))

        # Ratio scale (resize)
        x_ratio = resized_img_width / org_img_width
        y_ratio = resized_img_height / org_img_height

        # New bounding box of class table after resize
        xmin_table_resize = int(round(org_xmin_table * x_ratio))
        ymin_table_resize = int(round(org_ymin_table * y_ratio))
        xmax_table_resize = int(round(org_xmax_table * x_ratio))
        ymax_table_resize = int(round(org_ymax_table * y_ratio))

        # Bounding box of cropped image
        xmin_crop = xmin_table_resize - PADDING_TABLE
        ymin_crop = ymin_table_resize - PADDING_TABLE
        xmax_crop = xmax_table_resize + PADDING_TABLE
        ymax_crop = ymax_table_resize + PADDING_TABLE

        # Check limit condition
        xmin_crop = 0 if xmin_crop < 0 else xmin_crop
        ymin_crop = 0 if ymin_crop < 0 else ymin_crop
        xmax_crop = resized_img_width if xmax_crop > resized_img_width else xmax_crop
        ymax_crop = resized_img_height if ymax_crop > resized_img_height else ymax_crop

        # Cropped image
        xy_coordinates_org_table = (xmin_crop, ymin_crop, xmax_crop, ymax_crop)
        crop_img = resized_img.crop(xy_coordinates_org_table)
        crop_img.save(join_path(output_tsr_dir, image_name + '.jpg'))

        # AUTOMATIC IMAGE LABELING
        tsr_root = ET.Element("annotation")
        ET.SubElement(tsr_root, "filename").text = image_name + '.jpg'
        ET.SubElement(tsr_root, "folder").text = 'TSR PROJECT HOON'
        ET.SubElement(tsr_root, "path").text = join_path(output_tsr_dir, image_name + '.jpg')
        source = ET.SubElement(tsr_root, "source")
        ET.SubElement(source, "database").text = "Unknown"
        size = ET.SubElement(tsr_root, "size")
        ET.SubElement(size, "width").text = str(xmax_crop - xmin_crop)
        ET.SubElement(size, "height").text = str(ymax_crop - ymin_crop)
        ET.SubElement(size, "depth").text = "3"

        RATIO = [x_ratio, y_ratio]
        ORGINAL_PADDING = [ORIGINAL_PADDING_X, ORIGINAL_PADDING_Y]
        x_left_padding = xmin_table_resize - xmin_crop
        y_top_padding = ymin_table_resize - ymin_crop
        NEW_PADDING = [x_left_padding, y_top_padding]
        PADDING = [ORGINAL_PADDING, NEW_PADDING]

        automatic_recalculate_bbox_class_for_tsr(tsr_root, table, 'table', RATIO, PADDING)

        # Get original table_row list
        org_table_row_list = get_class_list_from_root(org_tree, CLASS_TABLE_ROW)

        # Get original table_column list
        org_table_column_list = get_class_list_from_root(org_tree, CLASS_TABLE_COLUMN)

        # Get original table_projected_row_header list
        org_table_projected_row_header_list = get_class_list_from_root(org_tree, CLASS_TABLE_PROJECTED_ROW_HEADER)

        # Get original table_spanning_cell list
        org_table_spanning_cell_list = get_class_list_from_root(org_tree, CLASS_TABLE_SPANNING_CELL)
        
        # Get original table_column_header list
        org_table_column_header_list = get_class_list_from_root(org_tree, CLASS_TABLE_COLUMN_HEADER)

        for org_table_row in org_table_row_list:
            automatic_recalculate_bbox_class_for_tsr(tsr_root, org_table_row, 'table row', RATIO, PADDING)

        for org_table_column in org_table_column_list:
            automatic_recalculate_bbox_class_for_tsr(tsr_root, org_table_column, 'table column', RATIO, PADDING)

        for org_table_column_header in org_table_column_header_list:
            automatic_recalculate_bbox_class_for_tsr(tsr_root, org_table_column_header, 'table column header', RATIO, PADDING)

        for org_table_spanning_cell in org_table_spanning_cell_list:
            automatic_recalculate_bbox_class_for_tsr(tsr_root, org_table_spanning_cell, 'table spanning cell', RATIO, PADDING)
        
        for org_table_projected_row_header in org_table_projected_row_header_list:
            automatic_recalculate_bbox_class_for_tsr(tsr_root, org_table_projected_row_header, 'table projected row header', RATIO, PADDING)
    
        tsr_tree = ET.ElementTree(tsr_root)
        xml_dir = join_path(output_tsr_dir, image_name + '.xml')
        tsr_tree.write(xml_dir)
