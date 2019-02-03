from keras import applications as ka, layers as kl, models as km
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential

import tensorflowjs as tfjs
import matplotlib.pyplot as plt
from keras.optimizers import SGD

def getmodel(cls=2):
    base_model = ka.nasnet.NASNetMobile(weights='imagenet', pooling='avg')
    # inceptionresnetv2, mobilenets, nasnet , imagenet
    print(base_model.summary())
    x = kl.Dense(cls, activation='softmax')(base_model.get_layer('global_average_pooling2d_1').output)
    model = km.Model(base_model.input, x)
    return model
if __name__ == '__main__':
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = getmodel()
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['acc'],
                  )
    # model.load_weights('path.h5')

    datagen = ImageDataGenerator(preprocessing_function=ka.resnet50.preprocess_input,
                                                 rotation_range=0.2,
                 width_shift_range=0.1,
                 height_shift_range=0.1,
                 horizontal_flip=True,
                 vertical_flip=False,
)
    train_data = datagen.flow_from_directory('train', target_size=(224, 224), batch_size=32)
    test_data = datagen.flow_from_directory('valida'
                                            ''
                                            'tion', target_size=(224, 224), batch_size=32)
    history = model.fit_generator(train_data, epochs=10000, validation_data=test_data,steps_per_epoch=len(train_data)/32, validation_steps=len(test_data)/32)
    model.save_weights('fishv2.h5')
    tfjs.converters.save_keras_model(model, 'here')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
