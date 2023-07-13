import time

from utils import *
from sklearn.model_selection import train_test_split

# PARAMS CHANGE
RAW_DATA_FOLDER_NAME = 'RAW_DATA'
OUTPUT_DATA_FOLDER_NAME = 'Table_Type_CDKT'

TEST_SIZE = [0.095, 0.105]
VAL_SIZE = [0.095, 0.105]

# PARAMS FIXED
BASE_ADDR = os.getcwd()
RAW_DATA_DIR = join_path(BASE_ADDR, RAW_DATA_FOLDER_NAME)
OUTPUT_DATA_DIR = join_path(BASE_ADDR, OUTPUT_DATA_FOLDER_NAME)

def main():
    # START
    print('Program start !!!')
    start_time = time.time()

    # Get image list & label list
    image_list = get_sample_list_from_path(RAW_DATA_DIR, IMAGE_FORMAT)
    label_list = get_sample_list_from_path(RAW_DATA_DIR, LABEL_FORMAT)
    sample_list = []

    # Create or Refresh output folder
    make_dirs_or_format_dir(OUTPUT_DATA_DIR)
    TD_FOLDER, TSR_FOLDER = make_dirs_for_table_extraction(OUTPUT_DATA_DIR)
    OUTPUT_RAW_TD_DIR = join_path(TD_FOLDER, 'images')
    OUTPUT_RAW_TSR_DIR = join_path(TSR_FOLDER, 'images')

    # NORMALIZE TASK
    # Resize image height & create ground truth file for task Table Detection & Table Structure Recognition
    start_time_task_normalize = time.time()
    img_index = -1
    for image in image_list:
        img_index += 1
        base_name = os.path.basename(image)
        sample_name = os.path.splitext(base_name)[0]
        sample_list.append(sample_name)

        start_time_per_step = time.time()
        normalize_td_image_size(image, label_list[img_index], OUTPUT_RAW_TD_DIR)
        normalize_tsr_image_size(image, label_list[img_index], OUTPUT_RAW_TSR_DIR)
        end_time_per_step = time.time()
        total_time_per_step = end_time_per_step - start_time_per_step
        print('Times normalize per image {:05}: {:.5f} (s)'.format(img_index + 1, round(total_time_per_step, 5)))
    
    end_time_task_normalize = time.time()
    total_time_task_normalize = end_time_task_normalize - start_time_task_normalize
    mean_times_per_normalize_img = total_time_task_normalize / (img_index + 1)
    mean_normalize_img_per_sec = 1 / mean_times_per_normalize_img
    print('Average normalize times per image:', round(mean_times_per_normalize_img, 5), '(s)')
    print('Average normalize image create per second: {:.3f}'.format(round(mean_normalize_img_per_sec, 2)), '(img/s)')
    print('Total running time of task normalize:', round(total_time_task_normalize, 5), '(s)')

    # RANDOM SPLIT TASK
    start_time_task_random_split = time.time()
    random.shuffle(sample_list)
    test_size = random_differrence_ratio(TEST_SIZE)
    val_size = random_differrence_ratio(VAL_SIZE)
    train_size = 1 - test_size - val_size
    train_val, test = train_test_split(sample_list, test_size=test_size)
    train, val = train_test_split(train_val, test_size=val_size/train_size)

    DATA_SPLIT = [train, val, test]
    DATA_SPLIT_NAME = ['train', 'val', 'test']
    TD_DIR = [OUTPUT_RAW_TD_DIR, TD_FOLDER]
    TSR_DIR = [OUTPUT_RAW_TSR_DIR, TSR_FOLDER]

    split_file(DATA_SPLIT, DATA_SPLIT_NAME, TD_DIR, TSR_DIR)

    end_time_task_random_split = time.time()
    total_time_task_random_split = end_time_task_random_split - start_time_task_random_split
    print('Total running time of task random split:', round(total_time_task_random_split, 5), '(s)')

    end_time = time.time()
    total_time = end_time - start_time
    print('Total running time:', round(total_time, 5), '(s)')

    # END
    print('Program run successfully !!!')
    print("Copyright @ProjectHoon")

if __name__ == '__main__':
    main()
