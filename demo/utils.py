import sys


def capture_cmdline(params):
    if len(sys.argv) == 1:
        return params

    model = sys.argv[1]
    dataset = sys.argv[2]
    attack_name = sys.argv[3]

    params["model"] = model
    params["dataset"] = dataset
    params["attack_name"] = attack_name

    if attack_name == "isolating_attack":
        isolated_cid = params["target_participant"]["target_cid"]
        params["isolated_participant"]["isolated_cid"] = isolated_cid

    elif attack_name == "overfitting_attack":
        params["attacker_participant"]["attacker_local_epochs"] *= 2

    layer_indexes = params[model]["exploited_layer_indexes"]
    params["inference_model"]["exploited_layer_indexes"] = layer_indexes

    gradient_indexes = params[model]["exploited_gradient_indexes"]
    params["inference_model"]["exploited_gradient_indexes"] = gradient_indexes

    return params


def map_mia(attack_name, epoch, cid,
            server, client, attacker,
            params, logger):
    if attack_name == "local_passive_attack":
        local_passive_attack(epoch, cid, server,
                             client, attacker, params, logger)
    elif attack_name == "overfitting_attack":
        overfitting_attack(epoch, cid, server,
                           client, attacker, params, logger)
    elif attack_name == "global_passive_attack":
        global_passive_attack(epoch, cid, server,
                              client, attacker, params, logger)
    elif attack_name == "isolating_attack":
        isolating_attack(epoch, cid, server,
                         client, attacker, params, logger)


def local_passive_attack(epoch, cid, server,
                         client, attacker, params, logger):
    target_participant_config = params["target_participant"]
    target_cid = target_participant_config["target_cid"]
    target_fed_epoch = target_participant_config["target_fed_epoch"]

    attacker_participant_config = params["attacker_participant"]
    attacker_cid = attacker_participant_config["attacker_cid"]
    attacker_local_epochs = attacker_participant_config["local_epochs"]

    participant_config = params["participant"]
    batch_size = participant_config["batch_size"]
    client_local_epochs = participant_config["local_epochs"]

    if epoch == target_fed_epoch and cid == target_cid:
        print("train inference model on victim (cid): {} "
              "at federated learning epoch: {}".format(target_cid, (target_fed_epoch + 1)))
        logger.info("train inference model on victim (cid): {}, "
                    "federated training epoch: {}".format(target_cid, (target_fed_epoch + 1)))
        attacker.create_membership_inference_model(client)
        attacker.train_inference_model()
        attacker.test_inference_model(client)

    print("[federated learning epoch: {}, "
          "current participant(cid): {}]".format((epoch + 1), cid))
    logger.info("federated training epoch: {}, "
                "current participant (cid): {}".format((epoch + 1), cid))

    client.download_global_parameters(server.global_parameters)

    if cid == attacker_cid:
        client.train_local_model(batch_size=batch_size,
                                 local_epochs=attacker_local_epochs)
    else:
        client.train_local_model(batch_size=batch_size,
                                 local_epochs=client_local_epochs)

    current_local_parameters = client.upload_local_parameters()
    server.accumulate_local_parameters(current_local_parameters)


def overfitting_attack(epoch, cid, server,
                       client, attacker, params, logger):
    target_participant_config = params["target_participant"]
    target_cid = target_participant_config["target_cid"]
    target_fed_epoch = target_participant_config["target_fed_epoch"]

    attacker_participant_config = params["attacker_participant"]
    attacker_cid = attacker_participant_config["attacker_cid"]
    attacker_local_epochs = attacker_participant_config["local_epochs"]

    participant_config = params["participant"]
    batch_size = participant_config["batch_size"]
    client_local_epochs = participant_config["local_epochs"]

    if epoch == target_fed_epoch and cid == target_cid:
        print("train inference model on victim (cid): {} "
              "at federated learning epoch: {}".format(target_cid, (target_fed_epoch + 1)))
        logger.info("train inference model on victim (cid): {}, "
                    "federated training epoch: {}".format(target_cid, (target_fed_epoch + 1)))
        attacker.create_membership_inference_model(client)
        attacker.train_inference_model()
        attacker.test_inference_model(client)

    print("[federated learning epoch: {}, "
          "current participant(cid): {}]".format((epoch + 1), cid))
    logger.info("federated training epoch: {}, "
                "current participant (cid): {}".format((epoch + 1), cid))

    client.download_global_parameters(server.global_parameters)

    if cid == attacker_cid:
        client.train_local_model(batch_size=batch_size,
                                 local_epochs=attacker_local_epochs)
    else:
        client.train_local_model(batch_size=batch_size,
                                 local_epochs=client_local_epochs)

    current_local_parameters = client.upload_local_parameters()
    server.accumulate_local_parameters(current_local_parameters)


def global_passive_attack(epoch, cid, server,
                          client, attacker, params, logger):
    target_participant_config = params["target_participant"]
    target_cid = target_participant_config["target_cid"]
    target_fed_epoch = target_participant_config["target_fed_epoch"]

    attacker_participant_config = params["attacker_participant"]
    attacker_cid = attacker_participant_config["attacker_cid"]
    attacker_local_epochs = attacker_participant_config["local_epochs"]

    participant_config = params["participant"]
    batch_size = participant_config["batch_size"]
    client_local_epochs = participant_config["local_epochs"]

    print("[federated learning epoch: {}, current participant (cid): {}]".format((epoch + 1), cid))
    logger.info("federated training epoch: {}, "
                "current participant (cid): {}".format((epoch + 1), cid))

    client.download_global_parameters(server.global_parameters)

    if cid == attacker_cid:
        client.train_local_model(batch_size=batch_size, local_epochs=attacker_local_epochs)
    else:
        client.train_local_model(batch_size=batch_size, local_epochs=client_local_epochs)

    current_local_parameters = client.upload_local_parameters()
    server.accumulate_local_parameters(current_local_parameters)

    if epoch == target_fed_epoch and cid == target_cid:
        print("train inference model on victim (cid): {} "
              "at federated learning epoch: {}".format(target_cid, (target_fed_epoch + 1)))
        logger.info("train inference model on victim (cid): {}, "
                    "federated training epoch: {}".format(target_cid, (target_fed_epoch + 1)))
        attacker.create_membership_inference_model(client)
        attacker.train_inference_model()
        attacker.test_inference_model(client)


def isolating_attack(epoch, cid, server,
                     client, attacker, params, logger):
    target_participant_config = params["target_participant"]
    target_cid = target_participant_config["target_cid"]
    target_fed_epoch = target_participant_config["target_fed_epoch"]

    attacker_participant_config = params["attacker_participant"]
    attacker_cid = attacker_participant_config["attacker_cid"]
    attacker_local_epochs = attacker_participant_config["local_epochs"]

    participant_config = params["participant"]
    batch_size = participant_config["batch_size"]
    client_local_epochs = participant_config["local_epochs"]

    print("[federated learning epoch: {}, current participant (cid): {}]".format((epoch + 1), cid))
    logger.info("federated training epoch: {}, "
                "current participant (cid): {}".format((epoch + 1), cid))

    client.download_global_parameters(server.global_parameters)

    if cid == attacker_cid:
        client.train_local_model(batch_size=batch_size, local_epochs=attacker_local_epochs)
    else:
        client.train_local_model(batch_size=batch_size, local_epochs=client_local_epochs)

    current_local_parameters = client.upload_local_parameters()
    server.accumulate_local_parameters(current_local_parameters)

    if epoch == target_fed_epoch and cid == target_cid:
        print("train inference model on victim (cid): {} "
              "at federated learning epoch: {}".format(target_cid, (target_fed_epoch + 1)))
        logger.info("train inference model on victim (cid): {}, "
                    "federated training epoch: {}".format(target_cid, (target_fed_epoch + 1)))
        attacker.create_membership_inference_model(client)
        attacker.train_inference_model()
        attacker.test_inference_model(client)
