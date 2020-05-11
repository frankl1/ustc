import os
import time
import pandas as pd
from test import *
from multiprocessing import Pool, cpu_count
from ust.transformers import UShapeletTransform
from ust.models import build_and_run_model
from ust.u_number import UNumber

SEED = 5813
coefs = ['0_1', '0_4' ,'0_6', '0_8', '1_0', '1_2', '1_4', '1_6', '1_8', '2_0']

time_limit_in_mins=10
datasets = ['ECG200', 'TwoLeadECG', 'ECGFiveDays', 'ItalyPowerDemand', 'SonyAIBORobotSurface2', 'SonyAIBORobotSurface1', 'Plane', 'Trace', 'SmoothSubspace', 'BME', 'CBF', 'UMD', 'SyntheticControl', 'ShapeletSim', 'Chinatown'] # 'DodgerLoopWeekend', 'DodgerLoopGame', 'DodgerLoopDay' contain NaN
cmps = [UNumber.SIMPLE_CMP, UNumber.CDF_CMP, UNumber.INTERVAL_CMP]
# cmps = [UNumber.SIMPLE_CMP]

# time_limit_in_mins=0.5
# coefs = ['0_1']
# datasets = ['Chinatown', 'BME']
# cmps = [UNumber.SIMPLE_CMP]
if __name__ == '__main__':
    cols = ['dataset', 'distance', 'cmp', 'acc', 'train_duration', 'test_duration']
    print('Numbers of datasets:', len(datasets))
    for use_ugnb in [True, False]:
        for coef in coefs:
            p = '-ugnb' if use_ugnb else ''
            # fname = f'results/result{p}-{coef}-{time.strftime("%Y%m%d")}.csv'
            fname = f'results/result-{p}-{coef}-{time.strftime("%Y%m%d")}.csv'
            scores = []
            args = []
            dataset_folder = os.path.join(os.getcwd(), "dataset", "Uncertain_Shapelet_ucr", coef)
            
            # difference cmp fo UED
            for cmp in cmps:
                args.extend([(d, dataset_folder, SEED, cmp, time_limit_in_mins, UShapeletTransform.UED, use_ugnb) for d in datasets])

            # adding dust
            if not use_ugnb:
                for dist in [UShapeletTransform.DUST_UNIFORM, UShapeletTransform.DUST_NORMAL, UShapeletTransform.ED]:
                    args.extend([(d, dataset_folder, SEED, UNumber.SIMPLE_CMP, time_limit_in_mins, dist, use_ugnb) for d in datasets]) 
            
            NB_PROCESS = min(len(args), cpu_count())
            
            print('Numbers of executions:', len(args))
            print('Numbers of workers:', NB_PROCESS)
            
            with Pool(NB_PROCESS) as pool:
                res = pool.starmap(build_and_run_model, args)
                scores.extend([[d, dist, cmp] + list(r) for (d,_,_,cmp,_,dist,_), r in zip(args, res)])
            df = pd.DataFrame(data=scores, columns=cols)
            df.to_csv(fname, index=False)
            print(f"scores coef{coef}:", scores)

    ## We run FOTS at the end because it is too slow
    print('Running FOTS')
    for coef in coefs:
        # fname = f'results/result{p}-{coef}-{time.strftime("%Y%m%d")}.csv'
        fname = f'results/result-fots-{coef}-{time.strftime("%Y%m%d")}.csv'
        scores = []
        dataset_folder = os.path.join(os.getcwd(), "dataset", "Uncertain_Shapelet_ucr", coef)
        use_ugnb = False
        args = [(d, dataset_folder, SEED, UNumber.SIMPLE_CMP, 12*time_limit_in_mins, UShapeletTransform.FOTS, use_ugnb) for d in datasets]
        
        NB_PROCESS = min(len(args), cpu_count())
                
        print('Numbers of executions:', len(args))
        print('Numbers of workers:', NB_PROCESS)
        
        with Pool(NB_PROCESS) as pool:
            res = pool.starmap(build_and_run_model, args)
            scores.extend([[d, dist, cmp] + list(r) for (d,_,_,cmp,_,dist,_), r in zip(args, res)])
        df = pd.DataFrame(data=scores, columns=cols)
        df.to_csv(fname, index=False)
        print(f"scores coef{coef}:", scores)