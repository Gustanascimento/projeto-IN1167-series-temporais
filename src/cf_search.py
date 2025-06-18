#!/usr/bin/env python
# coding: utf-8
import logging
import os
import pickle
import time

import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow as tf

from nbeats_keras.model import NBeatsNet
import tfts
import pmdarima


# plots
#import plotly.express as px
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import pandas as pd


os.environ["TF_DETERMINISTIC_OPS"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

warnings.filterwarnings(action="ignore", message="Setting attributes")

from forecastcf import ForecastCF, BaselineShiftCF, BaselineNNCF
# from _helper import (load_dataset, remove_extra_dim, add_extra_dim, DataLoader, MIMICDataLoader, forecast_metrics, cf_metrics)
from _helper_multi import (load_dataset, remove_extra_dim, add_extra_dim, DataLoader, MIMICDataLoader, forecast_metrics, cf_metrics)
from _utils import ResultWriter, reset_seeds


def main():
    parser = ArgumentParser(
        description="Run this script to evaluate ForecastCF method."
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset that the experiment is running on."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="The horizon of the forecasting task.",
    )
    parser.add_argument(
        "--back-horizon",
        type=int,
        required=True,
        help="The back horizon of the forecasting task",
    )
    parser.add_argument(
        "--split-size",
        nargs="+",
        type=float,
        default=[0.6, 0.2, 0.2],
        help="Split size for training/validation/testing following the temporal order, by default [0.6, 0.2, 0.2]",
    )
    parser.add_argument(
        "--stride-size",
        type=int,
        default=1,
        help="The sequence stride size when creating sequences from train/val/test, by default 1.",
    )
    parser.add_argument(
        "--center",
        type=str,
        default="median",
        help="The center parameter: the desired change's start value, by default 'median'.",
    )
    parser.add_argument(
        "--desired-shift",
        type=float,
        required=True,
        default=0,
        help="Desired shift value compared to the center, e.g., 0.2 (indicating 120% of the center value).",
    )
    parser.add_argument(
        "--desired-change",
        type=float,
        required=True,
        help="Desired increase/decrease trend parameter, e.g., 0.1, or -0.1",
    )
    parser.add_argument(
        "--poly-order",
        type=int,
        required=True,
        help="Desired poly function order, e.g., 1, 2, ...",
    )
    parser.add_argument(
        "--fraction-std",
        type=float,
        default=1,
        help="Fraction of standard deviation into creating the bound, e.g., 1, 1.5, 2, ...",
    )
    parser.add_argument(
        "--ablation-horizon",
        type=int,
        default=None,
        help="Ablation horizon parameter, for fixing the proportion of training sequences across different horizons.",
    )
    parser.add_argument(
        "--random-test",
        action="store_true",
        default=False,
        help="Boolean flag of using random 1000 samples from test set, default False.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=39,
        help="Random seed parameter, default 39.",
    )
    parser.add_argument(
        "--runtime-test",
        action="store_true",
        default=False,
        help="Boolean flag of runtime test using random 50 samples from test set, default False.",
    )
    parser.add_argument("--output", type=str, help="Output file name.")
    A = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")
    logger.info(f"Split size: {A.split_size}.")  # for debugging
    logger.info(f"Ablation horizon: {A.ablation_horizon}.")  # for debugging

    # desired starting center, desired shift, and desired change & fraction of std (bound widths)
    center, desired_shift = A.center.lower(), A.desired_shift
    poly_order = A.poly_order
    desired_change, fraction_std = A.desired_change, A.fraction_std

    logger.info(f"===========Desired trend parameters=============")
    logger.info(f"center: {center}, desired_shift: {desired_shift};")
    logger.info(f"fraction_std:{fraction_std};")
    logger.info(f"desired_change:{desired_change}, poly_order:{poly_order}.")

    RANDOM_STATE = A.random_seed
    result_writer = ResultWriter(file_name=A.output, dataset_name=A.dataset)

    logger.info(f"===========Random seed setup=============")
    logger.info(f"Random seed: {RANDOM_STATE}.")
    logger.info(f"Result writer is ready, writing to {A.output}...")
    # If `A.output` file already exists, no need to write head (directly append)
    if not os.path.isfile(A.output):
        result_writer.write_head()

    ###############################################
    # ## 1. Load data
    ###############################################
    data_path = "./data/"
    
    df = load_dataset(A.dataset, data_path)
    y = df.T.to_numpy()
    if y.ndim == 2:
        y = y[np.newaxis, :, :]
    n_features = y.shape[2]
    
    ablation_horizon = A.ablation_horizon
    back_horizon = A.back_horizon

    dataset = DataLoader(A.horizon, A.back_horizon)
    dataset.preprocessing_multi(
        y,
        n_features=n_features,
        train_size=A.split_size[0],
        val_size=A.split_size[1],
        normalize=True,
        sequence_stride=A.stride_size,
        ablation_horizon=A.ablation_horizon
    )

    
    logger.info(f"Data pre-processed, with {dataset.X_train.shape[0]} training samples, {dataset.X_val.shape[0]} validation samples, and {dataset.X_test.shape[0]} testing samples.")
    logger.info(f"Data shape: {dataset.X_train.shape}, {dataset.Y_train.shape}.")
    logger.info(f"Data shape: {dataset.X_val.shape}, {dataset.Y_val.shape}.")
    logger.info(f"Data shape: {dataset.X_test.shape}, {dataset.Y_test.shape}.")
    logger.info(f"Number of features: {n_features}.")
    
    # Ablation: Use `ablation_horizon`` as the horizon parameter after training/val/testing splits
    if ablation_horizon is not None:
        horizon = ablation_horizon

    # for model_name in ["sarimax","nbeats", "wavenet", "seq2seq", "gru"]:
    
    # a princípio testando somente GRU pra multivariada
    
    for model_name in ["gru"]:
        reset_seeds(RANDOM_STATE)
        
        if (model_name == "sarimax" and n_features > 1):
            logger.warning("SARIMAX model is not suitable for multivariate time series. Skipping this model.")
            continue

        ###############################################
        # ## 2.0 Forecasting model
        ###############################################

        if model_name in ["wavenet", "seq2seq"]:
            forecast_model = build_tfts_model(model_name, back_horizon, horizon)
            
        elif model_name == "nbeats":
            forecast_model = NBeatsNet(
                stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
                forecast_length=horizon,
                backcast_length=back_horizon,
                hidden_layer_units=256,
            )

            # Definition of the objective function and the optimizer
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
            forecast_model.compile(optimizer=optimizer, loss="mae")
            
        elif (model_name == "gru" and n_features == 1):
            forecast_model = tf.keras.models.Sequential(
                [
                    # Shape [batch, time, features] => [batch, time, gru_units]
                    tf.keras.layers.GRU(100, activation="tanh", return_sequences=True),
                    tf.keras.layers.GRU(100, activation="tanh", return_sequences=False),
                    # Shape => [batch, time, features]
                    tf.keras.layers.Dense(horizon, activation="linear"),
                    tf.keras.layers.Reshape((horizon, 1)),
                ]
            )
            # Definition of the objective function and the optimizer
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
            forecast_model.compile(optimizer=optimizer, loss="mae")
        
        elif model_name == "gru" and n_features > 1:
            # multivariáveis 
            
            forecast_model = tf.keras.Sequential()

            forecast_model.add(
                tf.keras.layers.GRU(
                    200,
                    activation="tanh",
                    input_shape=(back_horizon, n_features),
                    return_sequences=False  # último output do encoder
                )
            )

            forecast_model.add(tf.keras.layers.RepeatVector(horizon))  # repete para sequência de saída

            forecast_model.add(
                tf.keras.layers.GRU(
                    200,
                    activation="tanh",
                    return_sequences=True  # output sequência
                )
            )

            forecast_model.add(tf.keras.layers.Dropout(0.5))

            forecast_model.add(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(n_features, activation="linear")  # saída multivariada
                )
            )

            forecast_model.compile(
                loss="mean_squared_error",
                metrics=["mae", "mse"],
                optimizer=tf.keras.optimizers.Adam()
            )

            forecast_model.summary()

            
        elif model_name == "sarimax":
            logger.info("Using SARIMAX model for forecasting")

            model_pickle_path = f"{A.dataset}_sarimax_model.pkl"
            if os.path.exists(model_pickle_path):
                with open(model_pickle_path, "rb") as f:
                    forecast_model = pickle.load(f)
                logger.info(f"Loaded SARIMAX model from {model_pickle_path}")
            else:
                forecast_model = pmdarima.auto_arima(
                    dataset.X_train.reshape(-1, n_features)
                )
                with open(model_pickle_path, "wb") as f:
                    pickle.dump(forecast_model, f)
                logger.info(f"Trained and saved SARIMAX model to {model_pickle_path}")           
            
        else:
            print("Not implemented: model_name.")

        # Define the early stopping criteria
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
        )

        # Train the model
        reset_seeds(RANDOM_STATE)
        
        if model_name != "sarimax":
            forecast_model.fit(
                dataset.X_train,
                dataset.Y_train,
                epochs=100,
                batch_size=128,
                validation_data=(dataset.X_val, dataset.Y_val),
                callbacks=[early_stopping],
            )
            Y_pred = forecast_model.predict(dataset.X_test)
        else:
            # Predict on the testing set (forecast)
            Y_pred = forecast_model.predict(dataset.X_test.shape[0])
            
        mean_smape, mean_mase = forecast_metrics(dataset, Y_pred)

        logger.info(
            f"[[{model_name}]] model trained, with test sMAPE score {mean_smape:0.4f}; test MASE score: {mean_mase:0.4f}."
        )

        ###############################################
        # ## 2.1 CF search
        ###############################################
        cf_model = ForecastCF(
            max_iter=100,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            pred_margin_weight=0.25,
            step_weights=np.ones((1, dataset.X_train.shape[1], n_features)),
            random_state=RANDOM_STATE,
        )
        if model_name == "nbeats":
            cf_model.fit(forecast_model.models["forecast"], model_name)
        elif model_name in ["wavenet", "seq2seq", "gru", "sarimax"]:
            cf_model.fit(forecast_model, model_name)
        else:
            print("Not implemented: cf_model.fit.")

        # loss calculation ==> min/max bounds
        desired_max_lst, desired_min_lst = list(), list()

        if A.random_test == True:
            # use a subset of the test, especially for large dataset (e.g., M4)
            np.random.seed(RANDOM_STATE)
            rand_test_idx = np.random.choice(
                dataset.X_test.shape[0], 1000, replace=False
            )
        elif A.runtime_test == True:
            # use a subset of the test
            np.random.seed(RANDOM_STATE)
            rand_test_idx = np.random.choice(dataset.X_test.shape[0], 50, replace=False)
        else:
            rand_test_idx = np.arange(dataset.X_test.shape[0])

        X_test = dataset.X_test[rand_test_idx]
        Y_test = dataset.Y_test[rand_test_idx]
        logger.info(f"Generating CFs for {len(rand_test_idx)} samples in total...")
        for i in range(len(X_test)):
            # desired trend bounds: use the `center` parameter from the input sequence as the starting point
            # if horizon = 1: then desired_min_scaled = np.max(X_test[i]) * A.desired_min[0]
            desired_max_scaled, desired_min_scaled = generate_bounds(
                center=center,
                shift=desired_shift,
                change_percent=desired_change,
                poly_order=poly_order,
                horizon=horizon,
                fraction_std=fraction_std,
                input_series=X_test[i],
            )

            desired_max_lst.append(desired_max_scaled)
            desired_min_lst.append(desired_min_scaled)

        ###############################################
        # ## 2.2 runtime recording
        ###############################################
        start_time = time.time()
        cf_samples, losses, _ = cf_model.transform(
            X_test, desired_max_lst, desired_min_lst
        )
        end_time = time.time()
        elapsed_time1 = end_time - start_time
        logger.info(f"Elapsed time - ForecastCF: {elapsed_time1:0.4f}.")

        cf_model_bl = BaselineShiftCF(desired_percent_change=desired_change)
        start_time = time.time()
        cf_samples_bl = cf_model_bl.transform(X_test)
        end_time = time.time()
        elapsed_time2 = end_time - start_time
        logger.info(f"Elapsed time - BaseShift: {elapsed_time2:0.4f}.")

        cf_model_bl2 = BaselineNNCF()
        start_time = time.time()
        cf_model_bl2.fit(
            X_train=dataset.X_train.reshape((dataset.X_train.shape[0], dataset.X_train.shape[1] * dataset.X_train.shape[2])),
            Y_train=dataset.Y_train.reshape((dataset.Y_train.shape[0], dataset.Y_train.shape[1] * dataset.Y_train.shape[2]))
        )
        cf_samples_bl2 = cf_model_bl2.transform(desired_max_lst, desired_min_lst)
        end_time = time.time()
        elapsed_time3 = end_time - start_time
        logger.info(f"Elapsed time - BaseNN: {elapsed_time3:0.4f}.")

        ###############################################
        # ## 2.3 CF evaluation
        ###############################################
        input_indices = [range(0, back_horizon), range(0,n_features)]
        label_indices = range(back_horizon, back_horizon + horizon)
        cf_samples_lst = [cf_samples, cf_samples_bl, cf_samples_bl2]
        CF_MODEL_NAMES = ["ForecastCF", "BaseShift", "BaseNN"]

        for i in range(len(cf_samples_lst)):
            # predicted probabilities of CFs
            if model_name == "nbeats":
                z_preds = forecast_model.models["forecast"].predict(cf_samples_lst[i])
            elif model_name == "sarimax":
                    #size = int(((cf_samples_lst[i].size)/12))
                    z_preds=forecast_model.predict(cf_samples_lst[i].size)
                    z_preds=np.array([[z_preds[i:i+horizon] for i in range(0,len(z_preds),back_horizon)]])
                    z_preds=z_preds.reshape((z_preds.shape[1],z_preds.shape[2],1))                  
            else:
                z_preds = forecast_model.predict(cf_samples_lst[i])

            (
                validity,
                proximity,
                compactness,
                cumsum_valid_steps,
                cumsum_counts,
                cumsum_auc,
                #slope_diff,
                #slope_diff_preds,
            ) = cf_metrics(
                desired_max_lst,
                desired_min_lst,
                X_test,
                cf_samples_lst[i],
                z_preds,
                input_indices,
                label_indices,
            )
            
            # Plots
            
            #plot_horizon_test_graphs_plotly()
            #plot_ablation_study_graphs_plotly()
            

            logger.info(f"Done for CF search: [[{CF_MODEL_NAMES[i]}]].")
            logger.info(f"validity: {validity}, step_validity_auc: {cumsum_auc}.")
            logger.info(f"valid_steps: {cumsum_valid_steps}, counts:{cumsum_counts}.")
            logger.info(f"proximity: {proximity}, compactness: {compactness}.")
            # logger.info(
            #     f"slope_difference: {np.mean(slope_diff), np.std(slope_diff)}, slope_difference_preds:{np.mean(slope_diff_preds), np.std(slope_diff_preds)}."
            # )
            result_writer.write_result(
                random_seed=RANDOM_STATE,
                method_name=model_name,
                cf_method_name=CF_MODEL_NAMES[i],
                horizon=A.ablation_horizon,
                desired_change=desired_change,
                fraction_std=fraction_std,
                forecast_smape=mean_smape,
                forecast_mase=mean_mase,
                validity_ratio=validity,
                proximity=proximity,
                compactness=compactness,
                step_validity_auc=cumsum_auc,
            )
    logger.info("Done.")
    



    def plot_horizon_test_graphs_plotly(results_file='results/results_horizon_test.csv'):
        """
        Generates plots for the horizon test (Figure 3) using Plotly.
        """
        try:
            data = pd.read_csv(results_file)
        except FileNotFoundError:
            print(f"Error: The file {results_file} was not found.")
            return

        # Unpivot the dataframe to make it suitable for plotting with Plotly Express
        metrics_to_plot = ['validity_ratio', 'step_validity_auc', 'proximity', 'compactness']
        data_melted = data.melt(
            id_vars=['dataset', 'forecast_model', 'horizon'],
            value_vars=metrics_to_plot,
            var_name='metric',
            value_name='value'
        )

        # Create a faceted plot for each dataset
        fig = px.line(
            data_melted,
            x='horizon',
            y='value',
            color='forecast_model',
            line_dash='metric',
            facet_col='dataset',
            facet_col_wrap=3, # Adjust wrapping as needed
            title='Horizon Test by Dataset',
            labels={'value': 'Metric Value', 'horizon': 'Forecast Horizon'}
        )
        fig.update_yaxes(matches=None) # Allow y-axes to have different scales
        fig.show()


    def plot_ablation_study_graphs_plotly(results_file_cp='results/results_ablation_desired_change.csv', results_file_fr='results/results_ablation_fraction_std.csv'):
        """
        Generates plots for the ablation studies (Figures 4 & 5) using Plotly.
        """
        metrics = ['validity_ratio', 'step_validity_auc', 'proximity', 'compactness']
        models = ['n-beats', 'wavenet', 'seq2seq', 'gru']

        # --- Figure 4: Desired Change (cp) ---
        try:
            data_cp = pd.read_csv(results_file_cp)
            fig_cp = make_subplots(
                rows=2, cols=2,
                subplot_titles=metrics,
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )

            for i, metric in enumerate(metrics):
                for model in models:
                    model_data = data_cp[data_cp['forecast_model'] == model]
                    fig_cp.add_trace(
                        go.Scatter(x=model_data['desired_change'], y=model_data[metric], mode='lines', name=model),
                        row=(i // 2) + 1, col=(i % 2) + 1
                    )

            fig_cp.update_layout(
                title_text="Ablation Study: Desired Change Percent (cp)",
                height=700,
                showlegend=False
            )
            fig_cp.show()

        except FileNotFoundError:
            print(f"Error: The file {results_file_cp} was not found.")


        # --- Figure 5: Fraction of Standard Deviation (fr) ---
        try:
            data_fr = pd.read_csv(results_file_fr)
            fig_fr = make_subplots(
                rows=2, cols=2,
                subplot_titles=metrics,
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )

            for i, metric in enumerate(metrics):
                for model in models:
                    model_data = data_fr[data_fr['forecast_model'] == model]
                    fig_fr.add_trace(
                        go.Scatter(x=model_data['fraction_std'], y=model_data[metric], mode='lines', name=model),
                        row=(i // 2) + 1, col=(i % 2) + 1
                    )
            fig_fr.update_layout(
                title_text="Ablation Study: Fraction of Standard Deviation (fr)",
                height=700,
                showlegend=False
            )
            fig_fr.show()

        except FileNotFoundError:
            print(f"Error: The file {results_file_fr} was not found.")


def build_tfts_model(model_name, back_horizon, horizon):
    n_features = 1

    inputs = tf.keras.layers.Input([back_horizon, n_features])
    if model_name == "wavenet":
        backbone = tfts.AutoModel(
            model_name,
            predict_length=horizon,
            custom_model_params={
                "filters": 256,
                "skip_connect_circle": True,
            },
        )
    #     backbone = AutoModel("rnn", predict_length=horizon, custom_model_params = {'rnn_size': 256, "dense_size": 256})
    elif model_name == "seq2seq":
        backbone = tfts.AutoModel(
            "seq2seq",
            predict_length=horizon,
            custom_model_params={"rnn_size": 256, "dense_size": 256},
        )
    else:
        print("Not implemented: build_tfts_model.")
    outputs = backbone(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="mae")

    return model


def polynomial_values(shift, change_percent, poly_order, horizon):
    """
    shift: e.g., +0.1 (110% of the start value)
    change_percent: e.g., 0.1 (10% increase)
    poly_order: e.g., order 1, or 2, ...
    horizon: the forecasting horizon
    """
    if horizon == 1:
        return np.asarray([shift + change_percent])

    p_orders = [shift]  # intercept
    p_orders.extend([0 for i in range(poly_order)])
    p_orders[-1] = change_percent / ((horizon - 1) ** poly_order)

    p = np.polynomial.Polynomial(p_orders)
    p_coefs = list(reversed(p.coef))
    value_lst = np.asarray([[np.polyval(p_coefs, i) for x in range(8)] for i in range(horizon)])

    return value_lst


# TODO: apply limitation if the bound reach a desired value
def generate_bounds(
    center, shift, change_percent, poly_order, horizon, fraction_std, input_series
):
    if center == "last":
        start_value = input_series[-1, 0]
    elif center == "median":
        start_value = np.median(input_series)
    elif center == "mean":
        start_value = np.mean(input_series)
    elif center == "min":
        start_value = np.min(input_series)
    elif center == "max":
        start_value = np.max(input_series)
    else:
        print("Center: not implemented.")

    std = np.std(input_series)

    upper = start_value * (
            1
            + polynomial_values(shift, change_percent, poly_order, horizon)
            + fraction_std * std
        )
    
    lower =start_value * (
            1
            + polynomial_values(shift, change_percent, poly_order, horizon)
            - fraction_std * std
        )
    

    return upper, lower


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    main()
