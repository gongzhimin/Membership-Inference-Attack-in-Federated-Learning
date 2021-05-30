### Membership Inference Attack in Federated Learning ###
# To run background:
# nohup bash ./mia_boost.sh > mia_boost.log 2>&1 &

attack_name="passive local attack & passive global attack"
# passive local attack & passive global attack
python ./passive_attack_fed.py ${attack_name}

attack_name="isolating attack"
# isolating attack
python ./isolating_attack_fed.py ${attack_name}

attack_name="overfitting attack"
# overfitting attack
python ./overfitting_attack_fed.py ${attack_name}
