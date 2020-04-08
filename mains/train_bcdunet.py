from __future__ import division
from architectures.BCDU_net.model.BCDU_net import BCDU_net
from datasets.RetinaBloodVesselDataset import *
from architectures.BCDU_net.model.Preprocessing import *
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.layers import *
from keras import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    retina_blood_vessel_dataset = RetinaBloodVesselDataset()
    train_inputs, train_gt, train_bm = retina_blood_vessel_dataset.get_training_data()
    preprocessing = Preprocessing()
    prepro_inputs, prepro_bm = preprocessing.run_preprocess_pipeline(train_inputs, train_gt)
    #Using the einstein sum

    prepro_inputs = np.einsum('klij->kijl', prepro_inputs)
    prepro_bm = np.einsum('klij->kijl', prepro_bm)
    input_shape = (64,64,1)
    input = Input(input_shape)
    BCDU_NET = BCDU_net(input_shape)
    output = BCDU_NET(input)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='BCDU_NET_shapes.png', show_shapes=True, show_layer_names=True)
    model.summary()
    mcp_save = ModelCheckpoint('weight_lstm.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    model.fit(prepro_inputs, prepro_bm,
              batch_size=8,
              epochs=50,
              shuffle=True,
              verbose=1,
              validation_split=0.2, callbacks=[mcp_save, loss])
    print("sasda")

