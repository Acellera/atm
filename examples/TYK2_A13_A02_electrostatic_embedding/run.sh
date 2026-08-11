python -c "from atm.rbfe_structprep import rbfe_structprep; rbfe_structprep('QB_A13_A02_input.yaml')"
python -c "from atm.rbfe_production import rbfe_production; rbfe_production('QB_A13_A02_input.yaml')"
python -c "import yaml; from atm.uwham import calculate_uwham; c=yaml.safe_load(open('QB_A13_A02_input.yaml')); ddG,ddG_std,*_=calculate_uwham('.', c['BASENAME'], int(0.3*int(c['MAX_SAMPLES'])), int(c['MAX_SAMPLES'])); print('ddG:', ddG, '+/-', ddG_std, 'kcal/mol')"
