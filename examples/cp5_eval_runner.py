import socialsim as ss
import pandas as pd
import glob
import uuid

#Select the CP5 Configuration JSON file
config = './examples/data/cp5_configuration.json'
config = ss.load_config(config)

#Pass the CP5 Node List to metadata for evaluation
node_ls_fp = './examples/data/node_list.txt'
metadata = ss.MetaData(node_file= node_ls_fp)
cp5_nodes = [x.strip() for x in open(node_ls_fp,"r").readlines()]

#Select and load the ground truth weekly data then instantiate the task runner
path = './'
gt = glob.glob(path + '*.json')


ground_truth = []
for data in gt:
    print(data)
    ground_truth.append(ss.load_data(data, ignore_first_line=False, verbose=False))
ground_truth = pd.concat(ground_truth,sort=True)
ground_truth = ground_truth[ground_truth.informationID.isin(cp5_nodes)]

ground_truth = ground_truth[['informationID','nodeTime','nodeID','parentID','rootID','platform','actionType','nodeUserID','parentUserID']]

eval_runner = ss.EvaluationRunner(ground_truth, config, metadata=metadata)

submission_filepaths = []
for simulation_filepath in submission_filepaths:
    
    #Validation script in progress
    validation_flag = True
    if validation_flag:
        
        results, logs = eval_runner(simulation_filepath, verbose=True, submission_meta=True)
    else:
        print(validation_report)