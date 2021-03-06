import cv2
import copy
import random
import math



class Fill_color(object):
    def __init__(self, filename, label, cnt):
        self.file = ''
        self.start(filename, label, cnt)

    def start(self, file_name, label, cnt):
        origin_img = cv2.imread(file_name)
        if origin_img is None:
            print('====================== error - not found : ' + file_name + '======================')
            return

        gray_img = cv2.imread(file_name,0)
        bin_img = self.binarize(gray_img, 250)
        
        sg_img , count = self.segmentation(bin_img)
        result = self.segmentation_image_show(origin_img, sg_img, label, count, cnt)
        result = self.line_effect(sg_img, result, 7, 10)
        result = self.natual_coloring(result, 80)
        result = cv2.GaussianBlur(result, (3, 3), 0)
        cv2.imwrite('./multi_img_data/result/result.png', result)
        self.file = './multi_img_data/result/result.png'

    def binarize(self, img, threshold):
        # 이진화
        ret, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return bin_img

    def segmentation(self, img):
        segmentation_img = copy.deepcopy(img)
        offset = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        count = 0
        start_point = []
        
        for i in range(len(img)):
            for j in range(len(img[0])):
                if segmentation_img[i][j] == 255:
                    start_point.append([i,j])
                    count += 1  # segmentation_img[i][j] 가 255인 곳의 개수?

                    q = [[i,j]]
                    while q:
                        cur = q.pop(0)
                        x, y = cur[0], cur[1]
                        if x < 0 or y < 0:
                            pass
                        elif x > len(img) - 1 or y > len(img[0]) - 1: # out of bound
                            pass
                        elif segmentation_img[x][y] != 255:
                            pass
                        else:
                            segmentation_img[x][y] = count
                            for i in range(4):
                                q.append([x + offset[i][0], y + offset[i][1]])  # dfs 인듯?
        return [segmentation_img, count]


    def segmentation_image_show(self,origin_img, segmentation_img , label, count, cnt):
        color_img = copy.deepcopy(origin_img)
        # print(count) # 세그먼트 개수 출력
        # [4,2,173] # 체리색
        dic_label = {"apple":"사과","carrot":"당근","melon":"참외","strawberry":"딸기","tomato":"토마토","watermelon":"수박"}

        if cnt==1:  # 여러개의 색이 filling 되는 경우 한 번만 추정 label 을 출력
            print('\n\n=== 해당 이미지는 '+dic_label[label]+'(으)로 추정됩니다 === ')
        color_count = self.return_size(copy.deepcopy(segmentation_img),20)
        
        # !!!!!!!!!여기 수정해야함!!!!!!!!!!
        if label == 'apple':
            # 사과
            if cnt==1: color = [0,0,180]
            elif cnt ==2:color = [20,160,20]
            for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[0]:
                            color_img[i][j] = color 
        
        elif label == 'carrot':
            # 당근
            color = [[38,67,243],[10,180,10]]
            for seg_cnt in range(count - 1):
                for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[seg_cnt]:
                            if len(color)-1 <= seg_cnt:
                                color_img[i][j] = color[-1]
                            else:
                                color_img[i][j] = color[seg_cnt]
        elif label == 'melon':
            # 참외
            color = [0,220,220]
            for i in range(len(segmentation_img)):
                for j in range(len(segmentation_img[0])):
                    if segmentation_img[i][j] != 0 and segmentation_img[i][j] != 255 and segmentation_img[i][j] != 1:
                        color_img[i][j] = color
        elif label == 'strawberry':
            # 딸기
            color = [[54,54,255],[10,180,10]]
            for seg_cnt in range(count - 1):
                for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[seg_cnt]:
                            if len(color)-1 <= seg_cnt:
                                color_img[i][j] = color[-1]
                            else:
                                color_img[i][j] = color[seg_cnt]

        elif label == 'tomato':
            # 토마토
            color = [[0,0,180],[0,100,0]]
            for seg_cnt in range(count - 1):
                for i in range(len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[seg_cnt]:
                            if seg_cnt == 0:
                                color_img[i][j] = color[0]
                            else:
                                color_img[i][j] = color[1]
        elif label == 'watermelon':
            # 수박
            color = [20,160,20]
            start_point = 0
            black_list = []
            if count >= 3:
                for i in range(len(segmentation_img)):
                    check = False
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == 0:
                            start_point = i
                            # print(start_point)
                            check = True
                            break
                    if check:
                        break

            for seg_cnt in range(count - 1):
                for i in range(start_point, len(segmentation_img)):
                    for j in range(len(segmentation_img[0])):
                        if segmentation_img[i][j] == color_count[seg_cnt]:
                            black_list_check = True
                            # 블랙리스트 추가 조건
                            if start_point + 50 > i:
                                if not (segmentation_img[i][j] in black_list):
                                    black_list.append(segmentation_img[i][j])
                            # 블랙리스트면 색칠하지 않음
                            for k in range(len(black_list)):
                                if segmentation_img[i][j] == black_list[k]:
                                    black_list_check = False
                            if black_list_check:
                                color_img[i][j] = color
        return color_img

    def return_size(self,img, return_num):
        count_list = [0] * 255
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] != 255 and img[i][j] != 0 and img[i][j] != 1:
                    count_list[img[i][j]] += 1

        count_sort_list = []
        for i in range(return_num):
            count_sort_list.append(count_list.index(max(count_list)))
            count_list[count_sort_list[i]] = 0
        return count_sort_list

    def natual_coloring(self, img, value):
        # random_num = random.randrange(125,175)
        random_num = 125
        for i in range(random_num-value,random_num+value):
            for j in range(random_num-value,random_num+value):
                d = self.p2p_dst(i,j,random_num,random_num)
                if d <= value and self.img2np(img[i][j],[0,0,0]) and self.img2np(img[i][j],[255,255,255]):
                    for k in range(0,3):
                        img[i][j][k] = self.check255(img[i][j][k] + value - d)
        return img
        # img = cv2.GaussianBlur(img, (11, 11), 0)

    def p2p_dst(self,x1,y1,x2,y2):
        return int(math.sqrt((x2-x1)**2 + (y2-y1)**2))

    def img2np(self,v1,v2):
        if v1[0] == v2[0] and v1[1] == v2[1] and v1[2] == v2[2]:
            return False
        return True

    def check255(self,v):
        if v >= 255:
            return 255
        return v

    def line_effect(self, seg_img, color_img, value, n):
        for i in range(len(seg_img)):
            for j in range(len(seg_img[0])):
                for l in range(n):
                    if i + l > 298:
                        pass
                    else:
                        if seg_img[i][j] == 0 and seg_img[i+l][j] != 0 and seg_img[i+l][j] != 1:
                            for k in range(3):
                                if color_img[i+l][j][k] - value < 0:
                                    color_img[i+l][j][k] = 0
                                else:
                                    color_img[i+l][j][k] -= value
                    if j - l <= 0:
                        pass
                    else:
                        if seg_img[i][j] == 0 and seg_img[i][j-l] != 0 and seg_img[i][j-l] != 1:
                            for k in range(3):
                                if color_img[i][j-l][k] - value < 0:
                                    color_img[i][j-l][k] = 0
                                else:
                                    color_img[i][j-l][k] -= value
        return color_img
