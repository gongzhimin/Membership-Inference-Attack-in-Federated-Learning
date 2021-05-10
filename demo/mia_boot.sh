# model: alexnet | vgg16 | vgg19
# dataset: cifar10 | cifar-100 | cars
# attack_name: local_passive_attack | overfitting_attack | global_passive_attack | isolating_attack
# To run background:
# nohup bash ./demo/mia_boot.sh > mia.log 2>&1 &
model="alexnet"

####### Attack Features Comparison ######



###### Membership Inference Attack in Federated Learning ######
# CIFAR-10
dataset="cifar10"

attack_name="local_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="overfitting_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="global_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="global_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}


# CIFAR-100
dataset="cifar100"

attack_name="local_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="overfitting_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="global_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="global_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}


# Stanford-Cars
dataset="cars"

attack_name="local_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="overfitting_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="global_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}

attack_name="global_passive_attack"
python ./demo/mia_fed.py ${model} ${dataset} ${attack_name}
