from PIL import Image
import os, glob, numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2


class classification():
    def __init__(self):
        self.label = ''
        self.classify()

    def classify(self):
        caltech_dir = "./multi_img_data/imgs_others_test_sketch"

        image_w = 128
        image_h = 128

        pixels = image_h * image_w * 3

        X = []
        filenames = []
        files = glob.glob(caltech_dir+"/*.*")
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            filenames.append(f)
            X.append(data)
        plt.imshow(img)             # Resized image 를 출력
        plt.title("Resized Image")
        plt.show()
        X = np.array(X)
        model = load_model('./model/multi_img_classification.model')

        prediction = model.predict(X)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        cnt = 0

        # sg = Segmentation()

        for i in prediction:
            pre_ans = i.argmax()  # 예측레이블
            pre_ans_str = ''
            if pre_ans == 0: pre_ans_str = "사과"
            elif pre_ans == 1: pre_ans_str = "당근"
            elif pre_ans == 2: pre_ans_str = "참외"
            elif pre_ans == 3: pre_ans_str = "딸기"
            elif pre_ans == 4: pre_ans_str = "토마토"
            elif pre_ans == 5: pre_ans_str = "수박"

            if i[0] >= 0.8:
                # print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
                self.label = 'apple'
            elif i[1] >= 0.8:
                # print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
                self.label = 'carrot'
            elif i[2] >= 0.8:
                # print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
                self.label = 'melon'
            elif i[3] >= 0.8:
                # print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
                self.label = 'strawberry'
            elif i[4] >= 0.8:
                # print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
                self.label = 'tomato'
            elif i[5] >= 0.8:
                # print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
                self.label = 'watermelon'
            else:
                print("해당 이미지는 없는 데이터입니다.")
                self.label = 'none'


            cnt += 1
