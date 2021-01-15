import copy
import tensorflow as tf
from tensorflow.compat.v1.keras.utils import to_categorical

from membership_inference_attack.utils.attacker_utils import AttackerUtils
from membership_inference_attack.attacker_components.feature_extraction_cnn import *
from membership_inference_attack.attacker_components.feature_extraction_fcn import *
from membership_inference_attack.attacker_components.encoder import *
from membership_inference_attack.utils.losses import *

CNN_COMPONENT_LIST = ["Conv", "MaxPool"]
GRAD_LAYERS_LIST = ["Conv", "Dense"]


class MembershipInferenceAttack:
    def __init__(self,
                 target_model,
                 attacker_data_handler,
                 exploited_layer_indexes,
                 exploited_gradient_indexes,
                 exploit_label=True,
                 exploit_loss=True,
                 learning_rate=0.001,
                 epochs=10,
                 gradient_ascent=False):
        layers = target_model.layers
        AttackerUtils.sanity_check(layers, exploited_layer_indexes)
        AttackerUtils.sanity_check(layers, exploited_gradient_indexes)

        self.target_model = target_model
        self.attacker_data_handler = attacker_data_handler
        self.exploited_layer_indexes = exploited_layer_indexes
        self.exploited_gradient_indexes = exploited_gradient_indexes
        self.exploit_label = exploit_label
        self.exploit_loss = exploit_loss
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.gradient_ascent = gradient_ascent

        self.target_model_classes_num = int(target_model.output.shape[1])
        self.inference_model = None
        self.encoder = None

        # initialize input containers of inference model
        self.input_array = []
        self.attack_feature_tensors = []
        self.encoder_input_tensors = []

        # create attack features extraction components
        self.create_attack_features_extraction_components(layers)
        # create encoder
        self.encoder = create_encoder(self.encoder_input_tensors)
        # initialize inference model
        self.inference_model = tf.compat.v1.keras.Model(inputs=self.attack_feature_tensors, outputs=self.encoder)





    def create_layer_extraction_components(self, layers):
        for layer_index in self.exploited_layer_indexes:
            layer = layers[layer_index - 1]
            input_shape = layer.output_shape[1]
            cnn_needed = map(lambda i: i in layers.__class__.__name__, CNN_COMPONENT_LIST)
            if any(cnn_needed):
                layer_extraction_component = create_cnn_for_cnn_layer_outputs(layer.output_shape)
            else:
                layer_extraction_component = create_fcn_component(input_shape, 100)
            self.attack_feature_tensors.append(layer_extraction_component.input)
            self.encoder_input_tensors.append(layer_extraction_component.output)

    def create_label_extraction_component(self, output_size):
        label_extraction_component = create_fcn_component(output_size)
        self.attack_feature_tensors.append(label_extraction_component.input)
        self.encoder_input_tensors.append(label_extraction_component.output)

    def create_loss_extraction_component(self):
        loss_extraction_component = create_fcn_component(1, 100)
        self.attack_feature_tensors.append(loss_extraction_component.input)
        self.encoder_input_tensors.append(loss_extraction_component.output)

    def create_gradient_extraction_components(self, target_model, layers):
        gradient_layers = []
        for layer in layers:
            if any(map(lambda i: i in layer.__class__.__name__, GRAD_LAYERS_LIST)):
                gradient_layers.append(layer)
        variables = target_model.variables
        for layer_index in self.exploited_layer_indexes:
            layer = gradient_layers[layer_index - 1]
            gradient_shape = AttackerUtils.get_gradient_shape(variables, layer_index)
            cnn_needed = map(lambda i: i in layer.__class__.__name__, CNN_COMPONENT_LIST)
            if any(cnn_needed):
                gradients_extraction_component = create_cnn_for_cnn_gradients(gradient_shape)
            else:
                gradients_extraction_component = create_cnn_for_fcn_gradients(gradient_shape)
            self.attack_feature_tensors.append(gradients_extraction_component.input)
            self.encoder_input_tensors.append(gradients_extraction_component.output)

    def create_attack_features_extraction_components(self, layers):
        target_model = self.target_model

        # for layer outputs
        if self.exploited_layer_indexes and len(self.exploited_layer_indexes):
            self.create_layer_extraction_components(layers)

        # for one hot encoded labels
        if self.exploit_label:
            self.create_label_extraction_component(self.target_model_classes_num)

        # for loss
        if self.exploit_loss:
            self.create_loss_extraction_component()

        # for gradients
        if self.exploited_gradient_indexes and len(self.exploited_gradient_indexes):
            self.create_gradient_extraction_components(target_model, layers)

    def get_layer_outputs(self, target_model, features):
        layers = target_model.layers
        for layer_index in self.exploited_layer_indexes:
            target_model_input= target_model.input
            layer_output = layers[layer_index - 1].output
            hidden_layer_model = tf.compat.v1.keras.Model(target_model_input, layer_output)
            prediction = hidden_layer_model(features)
            self.input_array.append(prediction)

    def get_one_hot_encoded_labels(self, labels):
        one_hot_encoded_labels = to_categorical(labels, self.target_model_classes_num)
        return one_hot_encoded_labels

    def get_loss(self, target_model, features, labels):
        logits = target_model(features)
        loss = cross_entropy_loss(logits, labels)

        return loss

    def ascent_gradients_on_variables(self, gradients, variables):
        assert len(gradients) == len(variables), "gradients can't match to variables!"
        for (gradient, variable) in zip(gradients, variables):
            variable.assign_add(0.0001 * gradient)

    def compute_gradients(self, target_model, features, labels):
        split_features = AttackerUtils.split_variable(features)
        split_labels = AttackerUtils.split_variable(labels)
        gradients_array = []
        for (feature, label) in zip(split_features, split_labels):
            copied_target_model = copy.deepcopy(target_model)
            with tf.GradientTape() as tape:
                logits = copied_target_model(feature)
                loss = cross_entropy_loss(logits, label)
            target_variables = copied_target_model.variables
            gradients = tape.gradient(loss, target_variables)
            if self.gradient_ascent:
                self.ascent_gradients_on_variables(gradients, target_variables)
                gradients = tape.gradient(loss, target_variables)
            gradients_array.append(gradients)

    def get_gradients(self, target_model, features, labels):
        gradient_array = self.compute_gradients(target_model, features, labels)



