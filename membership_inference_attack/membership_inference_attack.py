import copy
import numpy as np
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
                 ascend_gradients=False):
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

        self.ascend_gradients = ascend_gradients

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
            target_model_input = target_model.input
            layer_output = layers[layer_index - 1].output
            hidden_layer_model = tf.compat.v1.keras.Model(target_model_input, layer_output)
            prediction = hidden_layer_model(features)
            self.input_array.append(prediction)

            layer_input_tensor = target_model.input
            prior_layer_output_values = features
            for layer in target_model.layers:
                layer_output_tensor = layer.output
                hidden_layer_model = tf.compat.v1.keras.Model(layer_input_tensor, layer_output_tensor)
                layer_output_values = hidden_layer_model(prior_layer_output_values)
                # craft the layer outputs
                prior_layer_output_values = layer_output_values
                layer_input_tensor = layer.output

    def get_one_hot_encoded_labels(self, labels):
        one_hot_encoded_labels = to_categorical(labels, self.target_model_classes_num)

        return one_hot_encoded_labels

    def get_loss(self, target_model, features, labels):
        logits = target_model(features)
        loss = cross_entropy_loss(logits, labels)

        return loss

    def ascend_gradients_on_variables(self, gradients, variables):
        assert len(gradients) == len(variables), "gradients can't match to variables!"
        for (gradient, variable) in zip(gradients, variables):
            variable.assign_add(0.0001 * gradient)

    def revert_variables(self, ascendant_variables, original_values):
        assert len(ascendant_variables) == len(original_values), "values can't match to variables!"
        for (value, variable) in zip(original_values, ascendant_variables):
            variable.assign(value)

    def compute_gradients(self, target_model, features, labels):
        split_features = AttackerUtils.split_variable(features)
        split_labels = AttackerUtils.split_variable(labels)
        gradients_array = [[]] * len(split_features)
        for index, (feature, label) in enumerate(zip(split_features, split_labels)):
            with tf.GradientTape() as tape:
                logits = target_model(feature)
                loss = cross_entropy_loss(logits, label)
            target_variables = target_model.variables
            copied_target_variables = copy.deepcopy(target_model.variables)
            gradients = tape.gradient(loss, target_variables)

            if self.ascend_gradients:
                self.ascend_gradients_on_variables(gradients, target_variables)
                with tf.GradientTape() as ascendant_tape:
                    logits = target_model(feature)
                    loss = cross_entropy_loss(logits, label)
                gradients = ascendant_tape.gradient(loss, target_variables)
                self.revert_variables(target_variables, copied_target_variables)

            gradients_array[index] = gradients

        return gradients_array

    def get_gradients(self, target_model, features, labels):
        gradients_array = self.compute_gradients(target_model, features, labels)
        gradients_batch = [[]] * len(gradients_array)
        for gradients_index, gradients in enumerate(gradients_array):
            gradient_per_layer = [[]] * len(self.exploited_gradient_indexes)
            for index, gradient_index in enumerate(self.exploited_gradient_indexes):
                gradient_index = (gradient_index - 1) * 2
                gradient_shape = gradients[gradient_index].shape
                reshaped = (int(gradient_shape[0]), int(gradient_shape[1]), 1)
                gradient = tf.reshape(gradients[gradient_index], reshaped)
                gradient_per_layer[index] = gradient
            gradients_batch[gradients_index] = gradient_per_layer

        gradients_batch = np.asarray(gradients_batch)
        split_gradients_batch = np.hsplit(gradients_batch, gradients_batch.shape[1])
        for split_gradients in split_gradients_batch:
            split_gradients_array = [[]] * len(split_gradients)
            for index in range(len(split_gradients)):
                split_gradients_array[index] = split_gradients[index][0]
            split_gradients_array = np.asarray(split_gradients_array)

            self.input_array.append(split_gradients_array)

    def get_gradient_norms(self, target_model, features, labels):
        gradients_array = self.compute_gradients(target_model, features, labels)
        gradients_batch = [[]] * len(gradients_array)
        for index, gradients in enumerate(gradients_array):
            gradients_batch[index] = np.linalg.norm(gradients[-1])

        return gradients_batch

    def forward_pass(self, target_model, features, labels):
        pass