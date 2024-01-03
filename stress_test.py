# # convert model
# python3 tools/export_model.py -c ./configs/rec/song_rec_svtrnet.yml -o Global.pretrained_model=./output/rec/svtr/best_accuracy  Global.save_inference_dir=./inference/rec_svtr_tiny_stn_en

# python3 tools/export_model.py -c ./configs/rec/PP-OCRv4/vn_PP-OCRv4_rec.yml -o Global.pretrained_model=../output/rec/rec_svtr_v4/best_accuracy  Global.save_inference_dir=../inference/rec_svtr_tiny_stn_vn_paddlev4
# predict
# python3 tools/infer/predict_rec.py --image_dir='./dataset/data_train_v1_2209/data_ocr_v1_2709/1137337.png' --rec_model_dir='./inference/rec_svtr_tiny_stn_en/' --rec_algorithm='SVTR' --rec_image_shape='3,32,480' --rec_char_dict_path='./dict/vn_dict.txt'

import cv2
import numpy as np
import time
import os
from tqdm import tqdm
from paddleocr import PaddleOCR

rec_char_dict_path='./dict/vn_dict.txt' 
class TextRecognitionPaddle:
    def __init__(self, device:int):
        if device < 0:
            self.ocr = PaddleOCR(use_gpu=False, rec_char_dict_path='./dict/plate.txt' , rec_algorithm='SVTR_LCNet', rec_image_shape='3,64,256', rec_model_dir='./inference/alpr/v2/', print=False)
        else:
            self.ocr = PaddleOCR(use_gpu=True, rec_char_dict_path='./dict/plate.txt' , rec_algorithm='SVTR_LCNet', rec_image_shape='3,64,256',  rec_model_dir='./inference/alpr/v3/', print=False)

        # print('>> [ PaddleOCR: MODE-GPU = {0}] loaded model paddleOCR'.format(use_gpu), flush=True)
    
    def recog(self, img):
        result = self.ocr.ocr(img, det=False, rec=True, cls=False)

        return result[0]

def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

def Convert_list(string):
    # Remove newline characters from the string
    cleaned_string = string.replace('\n', '')
    cleaned_string = cleaned_string.replace(' ', '')
    cleaned_string = cleaned_string.replace('-', '')
    cleaned_string = cleaned_string.replace('.', '')
    cleaned_string = cleaned_string.replace(',', '')

    # Convert the cleaned string to a list of characters
    list1 = []
    list1[:0] = cleaned_string
    return list1

def crop_center(img, padding = 0.1):
    # Read the image
    height, width = img.shape[:2]

    # Calculate 10% of each dimension
    left = int(width * padding)
    top = int(height * padding)
    right = int(width * (1 - padding))
    bottom = int(height * (1 - padding))

    # Perform the crop
    cropped_img = img[top:bottom, left:right]
    return cropped_img

if __name__ == '__main__':

    TextRecognition = TextRecognitionPaddle(device= 0)
    
    root = "/home/jovyan/sonnt373/data/ALPR_OCR/251123_train_plate_gen_250k"
    folder_txt = "/home/jovyan/sonnt373/data/ALPR_OCR/251123_train_plate_gen_250k/test1k.txt"
    CER = 0

    with open(folder_txt, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total_img = len(lines)
    t0 = time.time()
    acc_true = 0

    for line in tqdm(lines):

        label_text = line.split('\t')[1]
        img_path = os.path.join(root, line.split('\t')[0])

        # Load the image
        img = cv2.imread(img_path)

        # Increase contrast 
        # alpha = 1.5  # Contrast control (1.0-3.0)
        # beta = 0    # Brightness control (0-100)
        # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Get the aspect ratio of the image
        aspect_ratio = img.shape[1] / img.shape[0]

        # Calculate new width to maintain the same aspect ratio
        new_width = int(aspect_ratio * 200)

        # Resize the image to have a height of 200 while maintaining the aspect ratio
        resized_img = cv2.resize(img, (256, 64))
        # cv2.imwrite("check.jpg", resized_img)
        # resized_img = img

        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        # resized_img = cv2.filter2D(resized_img, -1, kernel)
        # resized_img = cv2.medianBlur(resized_img, 5)

        # Ordinary thresholding the same image
        # _, resized_img = cv2.threshold(resized_img, 155, 255, cv2.THRESH_BINARY)

        # if new_width > 2000:
        # #     # Get the amount of padding needed to reach a width of 480 pixels
        # pad_left = (1000 - new_width) // 2
        # pad_right = 1000 - new_width - pad_left

        # if pad_left > 0:
        #     # Pad the resized image to reach a size of 32x480
        #     resized_img = cv2.copyMakeBorder(resized_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        #     else:
        #         padded_img = cv2.resize(resized_img, (3000, 200))
        # else:
        #     padded_img = cv2.resize(resized_img, (400, 200))
        # Now, you can pass padded_img to the OCR
        # print("resized_img", resized_img.shape)
        # exit()
        # resized_img = crop_center(resized_img, 0.0)
        # cv2.imwrite("test.jpg", resized_img)
        result = TextRecognition.recog(resized_img)
        result_text = Convert_list(result[0][0])
        label_text_gt = Convert_list(label_text[:-1])

        if result_text == label_text_gt:
            acc_true += 1
        # else:
        #     cv2.imwrite(os.path.join( "./Check_false" , f"{label_text[:-1]}_{result[0][0]}.png" ), resized_img)

        cer = edit_distance(result_text, label_text_gt) / len(label_text_gt)

        # if cer > 0.1:
        #     acc_true += 1

        # print(f'==CER = {cer}====>>>>>predict is: {result[0][0]} ========> GT = {label_text}', new_width)
            # cv2.imwrite(os.path.join( "./Check_false" ,f'{label_text}.png'), padded_img)
        CER+=cer
    t_end = time.time()
    print(f"==Time per img = {(t_end - t0)/total_img}")
    print('Acc = ', acc_true/total_img)
    print("CER = ", CER/total_img)
