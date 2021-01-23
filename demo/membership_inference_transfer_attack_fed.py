import yaml

with open("demo\hyper_parameters.yaml", mode='r', encoding="utf-8") as f:
    hyper_parameters = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    dataset_info = hyper_parameters["dataset"]
    input_shape = dataset_info["input_shape"]
    classes_num = dataset_info["classes_num"]

    participant_info = hyper_parameters["participant"]
    epochs = participant_info["epochs"]
    learning_rate = participant_info["learning_rate"]
    client_num = participant_info["clients_num"]
    client_ratio_per_round = participant_info["client_ratio_per_round"]

