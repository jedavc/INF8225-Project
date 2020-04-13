from __future__ import division
from architectures.BCDU_net.model.BCDU_net import BCDU_net
from architectures.BCDU_net.model.VisualizePredictions import *
from datasets.RetinaBloodVesselDataset import *
from architectures.BCDU_net.model.Preprocessing import *
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot as plt
from keras.layers import *
from keras import Model
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from architectures.BCDU_net.model.Evaluation import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    retina_blood_vessel_dataset = RetinaBloodVesselDataset()
    train_inputs, train_gt, train_bm = retina_blood_vessel_dataset.get_training_data()
    preprocessing = Preprocessing()
    train_prepro_inputs, train_prepro_bm = preprocessing.run_preprocess_pipeline(train_inputs, "train", train_gt)
    # train_prepro_inputs = np.load('patches_imgs_train.npy')
    # train_prepro_bm = np.load('patches_masks_train.npy')

    #Using the einstein sum
    train_prepro_inputs = np.einsum('klij->kijl', train_prepro_inputs)
    train_prepro_bm = np.einsum('klij->kijl', train_prepro_bm)
    input_shape = (64,64,1)
    input = Input(input_shape)
    BCDU_NET = BCDU_net(input_shape)
    output = BCDU_NET(input)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='../BCDU_models/BCDU_NET_architecture.png', show_shapes=True, show_layer_names=True)
    model.summary()
    file_name = "../BCDU_models/new_v_model.{epoch:02d}-{loss:.4f}--{accuracy:.4f}--{val_accuracy:.4f}--{val_loss:.4f}.hdf5"
    file_name_loss = "../BCDU_models/new_model.{epoch:02d}-{loss:.4f}--{accuracy:.4f}--{val_accuracy:.4f}--{val_loss:.4f}.hdf5"
    save_model = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    val_save_model = ModelCheckpoint(file_name_loss, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    mcp_save = ModelCheckpoint('weight_50_lstm.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    # model = load_model("../BCDU_models/new_model.01-0.12--0.95--0.95--0.14.hdf5")
    csv_logger = CSVLogger("../BCDU_models/model_history_log.csv", append=True)
    history = model.fit(train_prepro_inputs, train_prepro_bm,
              batch_size=8,
              epochs=50,
              shuffle=True,
              verbose=1,
              validation_split=0.2, callbacks=[csv_logger, val_save_model, save_model, mcp_save, reduce_LR])
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("../BCDU_models/accuracy_history.png")
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("../BCDU_models/loss_history.png")

    #EVALUATE THE MODEL
    evaluation = Evaluation()
    test_prepro_inputs, test_prepro_bm, new_h, new_w = evaluation.evaluation_data()
    model.load_weights('weight_50_lstm.hdf5')
    preds = model.predict(test_prepro_inputs, batch_size=16, verbose=1)
    np.save('epoch_50_predictions', preds)
    # preds = np.load("new_predictions.npy")
    # np.load("new_predictions.npy")
    predictions = preds.reshape(-1,1,64,64)
    print("predicted images size :")
    print(predictions.shape)
    visualize_predictions = VisualizePredicitons()
    images, predictions_images, gt = visualize_predictions.make_visualizable(predictions, new_h, new_w, evaluation, test_prepro_bm)
    # ====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
    # Verify predictions inside the field of view
    y_scores, y_true = visualize_predictions.field_fo_view(predictions_images, gt, evaluation.test_bm)  # returns data only inside the FOV
    print(y_scores.shape)

    # Get evaluation metrics
    evaluation.evaluation_metrics(y_true, y_scores)



    # # Visualize
    # fig, ax = plt.subplots(10, 3, figsize=[15, 15])
    #
    # for idx in range(10):
    #     ax[idx, 0].imshow(np.uint8(np.squeeze((images[idx]))))
    #     ax[idx, 1].imshow(np.squeeze(gt[idx]), cmap='gray')
    #     ax[idx, 2].imshow(np.squeeze(prediction_images[idx]), cmap='gray')
    #
    # plt.savefig(path_experiment + 'sample_results.png')