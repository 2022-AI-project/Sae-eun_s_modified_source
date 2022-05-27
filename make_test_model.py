from PIL import Image
import os, glob, numpy as np
from numpy import ndarray as nda
from sklearn.model_selection import train_test_split

class make_model():
    def __init__(self):
        self.categories = ["apple", "carrot", "orientalmelon", "strawberry", "tomato", "watermelon"]
        self.nb_classes = len(self.categories)
        self.image_w = 128   # width of image
        self.image_h = 128   # height of image

        self.make_npy_file()
        self.make_model()

    
    # *.npy file 을 만드는 method 이다.
    def make_npy_file(self):
        # rotate 된 train data image 가 있는 directory
        caltech_dir_rotated = "./multi_img_data/imgs_others/train"

        X = []
        y = []

        for idx, cat in enumerate(self.categories):
            # 현재 label target score 를 정의한다.
            label = [0 for i in range(self.nb_classes)]
            label[idx] = 1

            image_dir_rotated = caltech_dir_rotated + "/" + cat
            files_rotated = glob.glob(image_dir_rotated + "/*.png")

            print("회전된", cat, "사진 개수 : ", len(files_rotated))
        
            for i, f in enumerate(files_rotated):
                img = Image.open(f)
                img = img.convert("RGB")
                img = img.resize((self.image_w, self.image_h))
                data = np.asarray(img)
               
                X.append(data)      # 현재 image data 를 append 한다.
                y.append(label)     # 현재 image 의 target score 를 append 한다.

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y)   # train, validation set으로 나뉜다.
        xy = (X_train, X_test, y_train, y_test)

        np.save("./numpy_data/multi_image_data_test.npy", xy)

    def make_model(self):
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        import matplotlib.pyplot as plt
        import keras.backend.tensorflow_backend as K
        import tensorflow as tf

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        X_train, X_test, y_train, y_test = np.load("./numpy_data/multi_image_data_test.npy", allow_pickle = True)
        
        X_train = X_train.astype(float) / 255
        X_test = X_test.astype(float) / 255

        '''
        1. 사용된 parameter 의 총 개수를 얻어보자.
        '''
        with K.tf_ops.device('/device:CPU:0'):
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
            #model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
            #model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.nb_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model_dir = './model'

            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            model_path = model_dir + '/multi_img_classification_test_(batch size).model'
            checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                         verbose=1, save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=6)

        model.summary()
        # batch size=32, 64, 128, 256, + 512
        history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])
        # batch_size, epochs 조절해가면서 변화 확인
        print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))

        y_vloss = history.history['val_loss']
        y_loss = history.history['loss']
        
        y_vacc = history.history['val_accuracy']
        y_acc = history.history['accuracy']

        x_len = np.arange(len(y_loss))

        # Loss function 과 Accuracy function 을 출력한다.
        plt.figure(1)
        plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
        plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid()
        
        plt.figure(2)
        plt.plot(x_len, y_vacc, marker='.', c='red', label='val_set_acc')
        plt.plot(x_len, y_acc, marker='.', c='blue', label='train_set_acc')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.grid()
        
        plt.show()
        
if __name__ == '__main__':
    w = make_model()
