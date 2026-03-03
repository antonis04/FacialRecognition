import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


def create_model(input_shape=(224, 224, 3)):
    """
    Tworzy model do klasyfikacji płci na bazie MobileNetV2.
    """
    # Bazowy model z wagami ImageNet, bez górnej warstwy klasyfikacyjnej
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Zamrażamy bazowy model – na początku nie będziemy trenować jego wag
    base_model.trainable = False

    # Dodajemy własne warstwy na górze
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # zmniejszamy wymiary
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # regularyzacja
    output = Dense(1, activation='sigmoid')(x)  # 1 neuron: 0 – kobieta, 1 – mężczyzna

    model = Model(inputs=base_model.input, outputs=output)
    return model