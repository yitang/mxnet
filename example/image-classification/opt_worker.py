from optomatic.worker import Worker
# from optomatic import Worker
from sklearn.cross_validation import cross_val_score
import numpy as np
# import user
import argparse
import logging
import yaml
import pandas as pd
from os import system
logger = logging.getLogger(__name__)


def parse_cli():
    parser = argparse.ArgumentParser(
        description='Get new parameters from database and compute their corresponding score.')
    parser.add_argument('--configure',
                        # default='27017',
                        required=True,
                        help='project configuration file.')
    parser.add_argument('--batch-mode',
                        '-b',
                        action='store_true',
                        help="write in batch mode, i.e. exit when there's no jobs")

    args = parser.parse_args()
    args.loop = not args.batch_mode
    return args


def objective_func(clf_params):
    print(clf_params)
    clf_params['lr'] = 10 ** clf_params['lr']
    clf_params['num_epochs'] = 5 * clf_params['num_epochs']
    clf_params['batch_size'] = 2 ** clf_params['batch_size']

    print(clf_params)
    # # clf.set_params(**clf_params)
    cmd0 = 'python3 tune_mnist.py \
    --lr %f \
    --num-epochs %d \
    --batch-size %d \
    --log-file tmp.log' % (clf_params['lr'], clf_params['num_epochs'], clf_params['batch_size'])
    print("Learning, run: ", cmd0)
    exe = system(cmd0)
    if exe != 0:
        raise ValueError

    cmd1 = 'Rscript parse_mxnet_log.R tmp.log'
    print("Processing log files, run ", cmd1)
    exe = system(cmd1)
    if exe != 0:
        raise ValueError
    system("rm tmp.log")

    train_score = pd.read_csv("tmp_train.csv")
    val_score = pd.read_csv("tmp_validation.csv")
    # scores = val_score
    # scores = cross_val_score(clf, X, y, scoring='log_loss', cv=4, n_jobs=-1)
    # scores = -1 * np.array(scores)
    # return [0, 0, 0], {'train_score': train_score.to_dict('records'),
    #                    'val_score': val_score.to_dict('records')}
    return [0], [train_score.to_dict('records'),val_score.to_dict('records')]  # json.loads(train_score.T.to_json()).values()
                 # 'val_score': json.loads(val_score.T.to_json()).values()}


def main():
    args = parse_cli()
    
    with open(args.configure, 'r') as f:
        config = yaml.load(f)

    for clf_name, db_collection in config['experiment_name'].items():
        # clf = user.clfs[clf_name]
        # w = Worker(config['project_name'], db_collection,
        #            objective_func, host=config['MongoDB']['host'], port=config['MongoDB']['port'],
        #            loop_forever=args.loop)
        w = Worker(config['project_name'], db_collection,
                   objective_func, host=config['MongoDB']['host'], port=config['MongoDB']['port'],
                   loop_forever=False)

        w.start_worker()
main()


# w.jobsDB.collection.remove()()
