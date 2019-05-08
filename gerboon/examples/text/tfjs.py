import tensorflowjs as tfjs
import keras as ks

model = ks.models.load_model("best_model_goed.hdf5")
tfjs.converters.save_keras_model(model, "best_model_goed.json")
