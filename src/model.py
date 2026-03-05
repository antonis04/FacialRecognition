import sys
import tensorflow as tf
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
from src.features import FEATURE_NAMES

def create_multimodal_model(input_shape=(224, 224, 3), num_features=len(FEATURE_NAMES)):
    image_input = Input(shape=input_shape, name='image_input')
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=image_input)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    image_features = x

    feature_input = Input(shape=(num_features,), name='feature_input')
    f = Dense(64, activation='relu')(feature_input)
    f = Dropout(0.3)(f)
    f = Dense(32, activation='relu')(f)

    combined = Concatenate()([image_features, f])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid', name='output')(z)

    model = Model(inputs=[image_input, feature_input], outputs=output)
    model.base_model = base_model
    return model