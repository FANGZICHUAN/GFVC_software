import sys
import os
import time
import yaml 

enc_configs = []
dec_configs = []
#=================================================================#
# for config in os.listdir('../test_all_config'):
#     if 'enc' in config:
#         # print(config)
#         enc_configs.append(config)
#     if 'dec' in config:
#         # print(config)
#         dec_configs.append(config)

# for config in enc_configs:
#     print('######'+config+'######')
#     config_path = '../test_all_config/'+config
#     os.system("python run.py --config "+ config_path )

# for config in dec_configs:
#     print('######'+config+'######')
#     config_path = '../test_all_config/'+config
#     os.system("python run.py --config "+ config_path )
#=================================================================#
for config in os.listdir('../test_all_trans_config'):
    if 'enc' in config:
        # print(config)
        enc_configs.append(config)
    if 'dec' in config:
        # print(config)
        dec_configs.append(config)
for config in dec_configs:
    print('######'+config+'######')
    config_path = '../test_all_trans_config/'+config
    os.system("python run.py --config "+ config_path )