# To run background:
# nohub bash ./mia_boost.sh > mia_boost.log 2>&1 &

nohub python ./passive_attack_fed.py > passive_attack_fed.log 2>&1 &

nohub python ./isolating_attack_fed.py > isolating_attack_fed.log 2>&1 &

nohub python ./overfitting_attack_fed.py > overfitting_attack_fed.log 2>&1 &

