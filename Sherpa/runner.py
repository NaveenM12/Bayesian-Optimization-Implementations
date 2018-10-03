import sherpa.sherpa1.core
import sherpa.sherpa1.schedulers
import sherpa.sherpa1.algorithms

parameters = [
              sherpa.sherpa1.core.Continuous('num_units', [100, 150]),
              ]

'''
parameters = [
              sherpa.sherpa1.core.Choice('num_conv_filters', [22, 32, 42]),
              #sherpa.Continuous('num_conv_filters', [25, 35]),
              sherpa.sherpa1.core.Choice('num_rows_conv_kernel', [1, 3, 5]),
              sherpa.sherpa1.core.Choice('num_cols_conv_kernel', [1, 3, 5]),
              sherpa.sherpa1.core.Choice('num_conv_filters2', [22, 32, 42]),
              sherpa.sherpa1.core.Choice('num_rows_conv_kernel2', [1, 3, 5]),
              sherpa.sherpa1.core.Choice('num_cols_conv_kernel2', [1, 3, 5]),
              sherpa.sherpa1.core.Choice('pooling_filter_w', [1,2, 5]),
              sherpa.sherpa1.core.Choice('pooling_filter_l', [1, 2, 5]),
              sherpa.sherpa1.core.Choice('num_units', [100, 128, 150]),
              ]
'''

alg = sherpa.sherpa1.algorithms.BayesianOptimization(max_num_trials=3)
scheduler = sherpa.sherpa1.schedulers.LocalScheduler()


sherpa.sherpa1.core.optimize(parameters=parameters,
                algorithm=alg,
                lower_is_better=True,  # Minimize objective
                filename='./trial.py',  # Python script to run, where the model was defined
                scheduler= scheduler,  # Run on local machine
                )

