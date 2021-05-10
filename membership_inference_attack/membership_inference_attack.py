import tensorflow as tf
from contextlib import redirect_stdout
from sklearn.metrics import accuracy_score

from membership_inference_attack.utils.attacker_utils import AttackerUtils
from membership_inference_attack.attacker_components.feature_extraction_cnn import *
from membership_inference_attack.attacker_components.feature_extraction_fcn import *
from membership_inference_attack.attacker_components.encoder import *
from membership_inference_attack.utils.losses import *
from membership_inference_attack.utils.attacker_optimizers import *
from membership_inference_attack.utils.visualization import *

CNN_COMPONENT_LIST = ["Conv", "MaxPool"]
GRAD_LAYERS_LIST = ["Conv", "Dense"]


class MembershipInferenceAttack:
    def __init__(self,
                 local_model,
                 attacker_data_handler,
                 exploited_layer_indexes,
                 exploited_gradient_indexes,
                 exploit_label=True,
                 exploit_loss=True,
                 learning_rate=0.001,
                 epochs=10,
                 optimizer_name="adam",
                 attack_msg=None,
                 logger=None,
                 ascend_gradients=False):

        layers = local_model.layers
        AttackerUtils.sanity_check(layers, exploited_layer_indexes)
        AttackerUtils.sanity_check(layers, exploited_gradient_indexes)

        self.local_model = local_model
        self.attacker_data_handler = attacker_data_handler
        self.exploited_layer_indexes = exploited_layer_indexes
        self.exploited_gradient_indexes = exploited_gradient_indexes
        self.exploit_label = exploit_label
        self.exploit_loss = exploit_loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = generate_optimizer(optimizer_name, learning_rate)

        self.logger = logger
        self.ascend_gradients = ascend_gradients

        self.local_model_classes_num = int(local_model.output.shape[1])
        self.one_hot_encoding_matrix = AttackerUtils.create_one_hot_encoding_matrix(self.local_model_classes_num)

        # initialize input containers of inference model
        self.input_array = []
        self.attack_feature_tensors = []
        self.encoder_input_tensors = []

        # create attack features extraction components
        self.create_attack_features_extraction_components(layers)
        # create encoder
        self.encoder = create_encoder(self.encoder_input_tensors)
        # initialize inference model
        self.inference_model = tf.keras.Model(inputs=self.attack_feature_tensors, outputs=self.encoder)

        assert attack_msg != None, "No attack message"
        self.attack_msg = attack_msg
        filepath = "logs/{}/plots/".format(self.attack_msg.attack_type)
        self.visualizer = Visualizer(filepath=filepath, attack_msg=self.attack_msg)

        self.log_info()

    def log_info(self):
        self.logger.info("[membership inference model] exploit label: {}, "
                         "exploit loss: {}".format(self.exploit_label, self.exploit_loss))

        self.logger.info("[membership inference model] exploit layer indexes: {}, "
                         "exploit gradient indexes: {}".format(self.exploited_layer_indexes,
                                                               self.exploited_gradient_indexes))

        self.logger.info("[membership inference model] optimizer: {}, "
                         "learning rate: {}, "
                         "epochs: {}".format(self.optimizer._name, self.learning_rate, self.epochs))

        # self.logger.info("[membership inference model] inference model details: ")
        # filename = self.logger.root.handlers[0].baseFilename
        # with open(filename, "a") as f:
        #     with redirect_stdout(f):
        #         self.inference_model.summary()

    def create_layer_extraction_components(self, layers):
        for layer_index in self.exploited_layer_indexes:
            layer = layers[layer_index - 1]
            input_shape = layer.output_shape[1]
            cnn_needed = map(lambda i: i in layer.__class__.__name__, CNN_COMPONENT_LIST)
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
        for layer_index in self.exploited_gradient_indexes:
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
        target_model = self.local_model

        # for layer outputs
        if self.exploited_layer_indexes and len(self.exploited_layer_indexes):
            self.create_layer_extraction_components(layers)

        # for one hot encoded labels
        if self.exploit_label:
            self.create_label_extraction_component(self.local_model_classes_num)

        # for loss
        if self.exploit_loss:
            self.create_loss_extraction_component()

        # for gradients
        if self.exploited_gradient_indexes and len(self.exploited_gradient_indexes):
            self.create_gradient_extraction_components(target_model, layers)

    def generate_layer_outputs(self, target_model, features):
        layers = target_model.layers
        for layer_index in self.exploited_layer_indexes:
            target_model_input = target_model.input
            layer_output = layers[layer_index - 1].output
            hidden_layer_model = tf.keras.Model(target_model_input, layer_output)
            prediction = hidden_layer_model(features)
            self.input_array.append(prediction)

    def generate_one_hot_encoded_labels(self, labels):
        one_hot_encoded_labels = AttackerUtils.one_hot_encode(labels, self.one_hot_encoding_matrix)
        self.input_array.append(one_hot_encoded_labels)

    def compute_loss(self, target_model, features, labels):
        logits = target_model(features)
        loss = cross_entropy_loss(logits, labels)
        loss = tf.reshape(loss, (len(loss.numpy()), 1))
        self.input_array.append(loss)

    # def ascend_gradients_on_variables(self, gradients, variables):
    #     assert len(gradients) == len(variables), "gradients can't match to variables!"
    #     for (gradient, variable) in zip(gradients, variables):
    #         variable.assign_add(0.0001 * gradient)
    #
    # def revert_variables(self, ascendant_variables, original_values):
    #     assert len(ascendant_variables) == len(original_values), "values can't match to variables!"
    #     for (value, variable) in zip(original_values, ascendant_variables):
    #         variable.assign(value)

    def compute_gradients(self, target_model, features, labels):
        split_features = AttackerUtils.split_variable(features)
        split_labels = AttackerUtils.split_variable(labels)
        gradients_array = [[]] * len(split_features)
        for index, (feature, label) in enumerate(zip(split_features, split_labels)):
            with tf.GradientTape() as tape:
                logits = target_model(feature)
                loss = cross_entropy_loss(logits, label)
            target_variables = target_model.variables
            # copied_target_variables = copy.deepcopy(target_model.variables)
            gradients = tape.gradient(loss, target_variables)

            # if self.ascend_gradients:
            #     self.ascend_gradients_on_variables(gradients, target_variables)
            #     with tf.GradientTape() as ascendant_tape:
            #         logits = target_model(feature)
            #         loss = cross_entropy_loss(logits, label)
            #     gradients = ascendant_tape.gradient(loss, target_variables)
            #     self.revert_variables(target_variables, copied_target_variables)

            gradients_array[index] = gradients

        return gradients_array

    def generate_gradients(self, target_model, features, labels):
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

    def compute_gradient_norms(self, target_model, features, labels):
        gradients_array = self.compute_gradients(target_model, features, labels)
        gradients_batch = [[]] * len(gradients_array)
        for (index, gradients) in enumerate(gradients_array):
            gradients_batch[index] = np.linalg.norm(gradients[-1])

        return gradients_batch

    def forward_pass(self, target_model, features, labels):
        self.input_array = []

        # Get the intermediate layer computations
        if self.exploited_layer_indexes and len(self.exploited_layer_indexes):
            self.generate_layer_outputs(target_model, features)

        # Get the one-hot-encoded labels
        if self.exploit_label:
            self.generate_one_hot_encoded_labels(labels)

        # Get the loss value
        if self.exploit_loss:
            self.compute_loss(target_model, features, labels)

        # Get the gradients
        if self.exploited_gradient_indexes and len(self.exploited_gradient_indexes):
            self.generate_gradients(target_model, features, labels)

        attack_outputs = self.inference_model(self.input_array)
        return attack_outputs

    def compute_attack_accuracy(self, member_data_batches, nonmember_data_batches):
        attack_accuracy = tf.keras.metrics.Accuracy("attack_accuracy", dtype=tf.float32)
        target_model = self.local_model

        for (member_data_batch, nonmember_data_batch) in zip(member_data_batches, nonmember_data_batches):
            self.inference_model.reset_metrics()

            member_features, member_labels = member_data_batch
            nonmember_features, nonmember_labels = nonmember_data_batch

            member_probabilities = self.forward_pass(target_model, member_features, member_labels)
            nonmember_probabilities = self.forward_pass(target_model, nonmember_features, nonmember_labels)
            y_pred = tf.concat((member_probabilities, nonmember_probabilities), 0)

            member_ones = tf.ones(member_probabilities.shape, dtype=bool)
            nonmember_zeros = tf.zeros(nonmember_probabilities.shape, dtype=bool)
            y_true = tf.concat((member_ones, nonmember_zeros), 0)

            attack_accuracy(y_pred > 0.5, y_true)

        attack_accuracy_result = attack_accuracy.result()
        return attack_accuracy_result

    def set_one_hot_encoding_matrix(self, classes_num):
        pass

    def train_inference_model(self):
        assert self.inference_model, "Inference model hasn't initialized!"
        member_train_data_batches, nonmember_train_data_batches, \
            nonmember_train_features, nonmember_train_labels = self.attacker_data_handler.load_train_data_batches()

        target_model = self.local_model
        target_model_pred = target_model(nonmember_train_features)
        target_model_accuracy = accuracy_score(nonmember_train_labels, np.argmax(target_model_pred, axis=1))
        print("Target model test accuracy: ", target_model_accuracy)
        self.logger.info("[membership inference model] target model test accuracy: {}".format(target_model_accuracy))

        member_test_data_batches, nonmember_test_data_batches = self.attacker_data_handler.load_test_data_batches()
        # member_test_data_batches = AttackerUtils.generate_subtraction(member_train_data_batches,
        #                                                               member_test_data_batches,
        #                                                               self.attacker_data_handler.batch_size)
        # nonmember_test_data_batches = AttackerUtils.generate_subtraction(nonmember_train_data_batches,
        #                                                                  nonmember_test_data_batches,
        #                                                                  self.attacker_data_handler.batch_size)

        best_attack_accuracy = 0
        # metric_attack_accuracy = tf.keras.metrics.Accuracy("attack_accuracy", dtype=tf.float32)
        zipped = zip(member_train_data_batches, nonmember_train_data_batches)
        print("Train membership inference attack model.")
        self.logger.info("[membership inference model] train membership inference attack model")
        for epoch in range(self.epochs):
            for ((member_features, member_labels), (nonmember_features, nonmember_labels)) in zipped:
                with tf.GradientTape() as tape:
                    tape.reset()
                    member_outputs = self.forward_pass(target_model, member_features, member_labels)
                    nonmember_outputs = self.forward_pass(target_model, nonmember_features, nonmember_labels)

                    member_ones = tf.ones(member_outputs.shape)
                    nonmember_zeros = tf.zeros(nonmember_outputs.shape)

                    y_pred = tf.concat((member_outputs, nonmember_outputs), 0)
                    y_true = tf.concat((member_ones, nonmember_zeros), 0)
                    attack_loss = mse(y_true, y_pred)

                grads = tape.gradient(attack_loss, self.inference_model.variables)
                self.optimizer.apply_gradients(zip(grads, self.inference_model.variables))

            # metric_attack_accuracy(y_pred > 0.5, y_true)

            attack_accuracy = self.compute_attack_accuracy(member_test_data_batches, nonmember_test_data_batches)
            if attack_accuracy > best_attack_accuracy:
                best_attack_accuracy = attack_accuracy

            print("attack epoch {}: attack test accuracy: {}, best attack accuracy: {}".format((epoch + 1),
                                                                                               attack_accuracy,
                                                                                               best_attack_accuracy))
            self.logger.info("[membership inference model] attack epoch: {}, "
                             "attack test accuracy: {}, "
                             "best attack accuracy: {}".format((epoch + 1),
                                                               attack_accuracy,
                                                               best_attack_accuracy))

    def reset_target_model_visibility(self):
        pass

    def visually_test_inference_model(self, target_model, verifier_data_handler):
        member_target_data_batches, nonmember_target_data_batches, \
            nonmember_target_features, nonmember_target_labels = verifier_data_handler.load_target_data_batches()
        zipped = zip(member_target_data_batches, nonmember_target_data_batches)

        target_model_pred = target_model(nonmember_target_features)
        target_model_accuracy = accuracy_score(nonmember_target_labels, np.argmax(target_model_pred, axis=1))
        print("Target model test accuracy: {}".format(target_model_accuracy))
        self.logger.info("[membership inference model] target model test accuracy: {}".format(target_model_accuracy))

        member_true_list, nonmember_true_list = [], []
        member_preds_list, nonmember_preds_list = [], []
        member_features_list, member_labels_list = [], []
        nonmember_features_list, nonmember_labels_list = [], []
        member_gradient_norms_list, nonmember_gradient_norms_list = [], []

        for ((member_features, member_labels), (nonmember_features, nonmember_labels)) in zipped:
            member_preds = self.forward_pass(target_model, member_features, member_labels)
            member_gradient_norms = self.compute_gradient_norms(target_model, member_features, member_labels)

            member_preds_list.extend(member_preds.numpy())
            member_gradient_norms_list.extend(member_gradient_norms)
            member_features_list.extend(member_features)
            member_labels_list.extend(member_labels)
            member_true = np.ones(member_preds.shape)
            member_true_list.extend(member_true)

            nonmember_preds = self.forward_pass(target_model, nonmember_features, nonmember_labels)
            nonmember_gradient_norms = self.compute_gradient_norms(target_model, nonmember_features, nonmember_labels)

            nonmember_preds_list.extend(nonmember_preds.numpy())
            nonmember_gradient_norms_list.extend(nonmember_gradient_norms)
            nonmember_features_list.extend(nonmember_features)
            nonmember_labels_list.extend(nonmember_labels)
            nonmember_true = np.zeros(nonmember_preds.shape)
            nonmember_true_list.extend(nonmember_true)

        y_true = tf.concat((member_true_list, nonmember_true_list), 0)
        y_pred = tf.concat((member_preds_list, nonmember_preds_list), 0)

        attack_accuracy = tf.keras.metrics.Accuracy("attack_accuracy", dtype=tf.float32)
        attack_accuracy(y_pred > 0.5, y_true)
        attack_accuracy_result = attack_accuracy.result()
        print("Attack accuracy: {}".format(attack_accuracy_result))
        self.logger.info("[membership inference model] attack accuracy: {}".format(attack_accuracy_result))

        self.visualizer.plot_membership_probability_histogram(member_preds_list, nonmember_preds_list)
        self.visualizer.plot_membership_inference_attack_roc_curve(y_true, y_pred)
        self.visualizer.plot_gradient_norm_scatter(member_labels_list, member_gradient_norms_list,
                                                   nonmember_labels_list, nonmember_gradient_norms_list)
        self.visualizer.plot_per_label_membership_probability_histogram(member_labels_list, member_preds_list,
                                                                        nonmember_labels_list, nonmember_preds_list)
