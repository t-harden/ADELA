from tpot import TPOTClassifier
import numpy as np
import pickle
import time


with open('utils/dict_demo.pkl', 'rb') as f:
    dict_task_data = pickle.load(f)

task_id = 3
X, y = dict_task_data[task_id][0].get_X_and_y()
dataset_embed = dict_task_data[task_id][2]
dataset_embed = np.nan_to_num(dataset_embed)

est = TPOTClassifier(generations=5, population_size=10, verbosity=2, n_jobs=1, random_state=42, config_dict='TPOT light',
                     template='Selector-Selector-Transformer-Transformer-Classifier',
                     surrogate_accompany_model='trained_model.pt',
                     dataset_embed=dataset_embed)

st_time = time.time()
est.fit(X, y)
end_time = time.time()
final_performance = est.score(X, y)
print("Final sklearn pipeline: ", est.fitted_pipeline_)
print("Final performance: ", final_performance)
print("Time: ", end_time-st_time)

