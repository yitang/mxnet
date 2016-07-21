from optomatic.jobs import JobsDB
from optomatic.worker import Worker
import yaml
import pandas as pd

with open('opt_config_worker.yaml') as f:
    config = yaml.load(f)

metrics = {'train': [],
           'val': []}
exp_counter = 0
for clf_name, db_collection in config['experiment_name'].items():
    # clf = user.clfs[clf_name]
    # w = Worker(config['project_name'], db_collection,
    #            objective_func, host=config['MongoDB']['host'], port=config['MongoDB']['port'],
    #            loop_forever=args.loop)
    # w = Worker(config['project_name'], db_collection,
    #            None, host=config['MongoDB']['host'], port=config['MongoDB']['port'],
    #            loop_forever=False)
    conn = JobsDB(config['project_name'], db_collection,
                  host=config['MongoDB']['host'], port=config['MongoDB']['port'])
    exps = list(conn.collection.find())
    log_metrics = {'train': [],
                   'val': []}
    for exp in exps:
        res = exp['aux_data']
        if res:        
            t_m = pd.DataFrame(res[0])
            v_m = pd.DataFrame(res[1])
            exp_counter += 1
        else:
            # raise ValueError('Empty results for experiments', exp)
            print('Empty results for experiments', exp)
            continue
        params = exp['params']
        params['meta_params'] = clf_name
        params['exp_id'] = exp_counter
        for k, v in params.items():
            t_m.insert(0, k, v)
            v_m.insert(0, k, v)

        log_metrics['train'].append(t_m)
        log_metrics['val'].append(v_m)
        del t_m, v_m
    tmp = pd.concat((log_metrics['train']), axis=0)
    tmp.to_csv("train_%s.csv" % db_collection, index=False)    
    # metrics['train'].append(tmp)
    tmp = pd.concat((log_metrics['val']), axis=0)
    tmp.to_csv("val_%s.csv" % db_collection, index=False)    
    # metrics['val'].append(tmp)

    

