import sherpa.sherpa1.core
import sherpa.sherpa1.algorithms
import sherpa.sherpa1.schedulers

parameters = [sherpa.sherpa1.core.Choice('num_units', [100, 200, 300]),]
alg = sherpa.sherpa1.algorithms.RandomSearch(max_num_trials=150)
rval = sherpa.sherpa1.core.optimize(parameters=parameters,
                       algorithm=alg,
                       lower_is_better=True,  # Minimize objective
                       filename='./trialtest.py', # Python script to run, where the model was defined
                       scheduler=sherpa.sherpa1.schedulers.LocalScheduler(), # Run on local machine
                       )