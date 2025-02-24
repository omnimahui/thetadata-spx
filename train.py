import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import shutil

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

from pathlib import Path

from tqdm.autonotebook import tqdm
from IPython.display import display, HTML
# %load_ext autoreload
# %autoreload 2
np.random.seed(42)
tqdm.pandas()
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
pl.seed_everything(42)
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE, MAE
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import TemporalFusionTransformer

logger = TensorBoardLogger("tb_logs", name="China-stock")

# Make True to select a subsample. Helps with faster training.
TRAIN_SUBSAMPLE = False

def format_plot(fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        title_text=title,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        titlefont={"size": 20},
        legend_title=None,
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title_text=ylabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
        xaxis=dict(
            title_text=xlabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        )
    )
    return fig

from itertools import cycle

def highlight_abs_min(s, props=''):
    return np.where(s.abs() == np.nanmin(np.abs(s.values)), props, '')

from lightning.pytorch.callbacks import ModelCheckpoint

def load_weights(model, weight_path):
    state_dict = ModelCheckpoint(dirpath=weight_path)
    model.load_state_dict(state_dict)



    
from collections import namedtuple

FeatureConfig = namedtuple(
    "FeatureConfig",
    [
        "target",
        "index_cols",
        "static_categoricals",
        "static_reals",
        "time_varying_known_categoricals",
        "time_varying_known_reals",
        "time_varying_unknown_reals",
        "group_ids"
    ],
)

#Reading the missing value imputed and train test split data
#train_df = pd.read_parquet(preprocessed/"selected_blocks_train_missing_imputed_feature_engg.parquet")
# Read in the Validation dataset as test_df so that we predict on it
#test_df = pd.read_parquet(preprocessed/"selected_blocks_val_missing_imputed_feature_engg.parquet")
# test_df = pd.read_parquet(preprocessed/"selected_blocks_test_missing_imputed_feature_engg.parquet")

folder = os.path.dirname(os.path.realpath(__file__))+os.sep
#train_df = pd.read_hdf(folder+"thetadata-spx-features.hdf",key='features',mode="r")
train_df = pd.read_parquet(folder+"thetadata-spx-features.parquet")
#train_df=train_df.loc[(train_df.index.get_level_values(1) == '000001.SZ') | (train_df.index.get_level_values(1) == '000002.SZ')] 
train_df.reset_index(inplace=True)
train_df['pct_chg'] = train_df.groupby("date")['Price'].pct_change()
train_df["minute_of_day"] = pd.to_datetime(train_df["time"], format="%H:%M:%S").dt.hour * 60 + pd.to_datetime(train_df["time"], format="%H:%M:%S").dt.minute
#time_idx must be within date
train_df.dropna(inplace=True)
train_df["time_idx"] = train_df.groupby("date").cumcount()


feat_config = FeatureConfig(
    target="pct_chg",
    index_cols=["date", "time"],
    static_categoricals=[],  # Categoricals which does not change with time
    static_reals=[],  # Reals which does not change with time
    time_varying_known_categoricals=[],
    time_varying_known_reals=["minute_of_day"], 
    time_varying_unknown_reals=[  # Reals which change with time, but we don't have the future. Like the target
        "Garch_Vlty","Hist_Vlty",'atm_plus_0spread_iv_CALL',
       'atm_plus_0spread_iv_PUT'
    ],  
    group_ids=[  # Feature or list of features which uniquely identifies each entity
        "date"
    ],  
)

max_prediction_length = 10
min_prediction_length = 5
max_encoder_length = 40
min_encoder_length = 30
batch_size = 1024  # set this to a value which your GPU can handle
train_model = True # Set this to True to train model. Else will load saved models ! Warning! Training on full dataset takes 3-6 hours
tag = "SPX-minute-TFT"
metric_record = []
individual_metrics = dict()

TRAIN_PERIOD_END = "20240101"
val_df = train_df[(train_df.date >= TRAIN_PERIOD_END)]
train_df = train_df[train_df.date < TRAIN_PERIOD_END]

#df_test = df[df["date"] >= VAL_PERIOD_END]
print(f"Total train rows: {len(train_df)}, Total validation rows: {len(val_df)}, Total test rows: 0")

#pred_df = test_df[feat_config.index_cols+[feat_config.target]+['time_idx']].copy()
# pred_df.set_index(feat_config.index_cols, inplace=True)

cols = feat_config.index_cols + [feat_config.target]
full_df = pd.concat(
    [
        train_df[cols],
        val_df[cols],
    ]
).set_index(feat_config.index_cols)

# Defining the training dataset
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target=feat_config.target,
    group_ids=feat_config.group_ids,
    max_encoder_length=max_encoder_length,
    min_encoder_length=min_encoder_length,
    max_prediction_length=max_prediction_length,
    min_prediction_length=min_prediction_length,
    static_categoricals=feat_config.static_categoricals,
    static_reals=feat_config.static_reals,
    time_varying_known_categoricals=feat_config.time_varying_known_categoricals,
    time_varying_known_reals=feat_config.time_varying_known_reals,
    time_varying_unknown_reals=feat_config.time_varying_unknown_reals,
    #Not need since pct_chg is [0,1]
    #target_normalizer=GroupNormalizer(
    #    groups=feat_config.group_ids, transformation=None
    #),
    allow_missing_timesteps=True,
    #categorical_encoders={"symbol": NaNLabelEncoder(add_nan=True),
    #                      "sector": NaNLabelEncoder(add_nan=True)},
    #not need since time_idx is numeric from 0
    #add_relative_time_idx=True,
    #not need since constant group length.
    add_encoder_length=True,
)

# Defining the validation dataset with the same parameters as training
validation = TimeSeriesDataSet.from_dataset(training, pd.concat([val_df]).reset_index(drop=True), stop_randomization=True)
# Defining the test dataset with the same parameters as training
#test = TimeSeriesDataSet.from_dataset(training, pd.concat([hist_df, test_df]).reset_index(drop=True), stop_randomization=True)

#Added to fix categorical_encoders
training = TimeSeriesDataSet.from_dataset(training, train_df.reset_index(drop=True), stop_randomization=True)

# Making the dataloaders
# num_workers can be increased in linux to speed-up training
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


#cardinality = [len(training.categorical_encoders[c].classes_) for c in training.categoricals]
#embedding_sizes = {
#    col: (x, min(50, (x + 1) // 2))
#    for col, x in zip(training.categoricals, cardinality)
#}

model_params = {
    "hidden_size": 128,
    "lstm_layers": 2,
    "attention_head_size": 4,
    #"hidden_continuous_size": 128,
    #"dropout": 0.1,
    #"embedding_sizes": embedding_sizes
    #"embedding_sizes":{}
}
other_params = dict(
    learning_rate=1e-2,
    optimizer="adam",
    loss=RMSE(),
    logging_metrics=[RMSE(), MAE()],
)

model = TemporalFusionTransformer.from_dataset(
    training,**{**model_params, **other_params})
#Testing out the model
x, y = next(iter(train_dataloader))
_ = model(x)
type(_), _.prediction.shape

saved_model_sampled = f'{tag}_sampled.wt'
saved_model_full = f'{tag}.wt'

if train_model:
    trainer = Trainer(
        logger=logger,
        accelerator="cuda",
        min_epochs=1,
        max_epochs=20,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10 if TRAIN_SUBSAMPLE else 4*3),
            ModelCheckpoint(
                monitor="val_loss", save_last=True, mode="min", auto_insert_metric_name=True
            ),
        ],
        val_check_interval=1.0 if TRAIN_SUBSAMPLE else 0.5,
        log_every_n_steps=50 if TRAIN_SUBSAMPLE else 0.5,
        gradient_clip_val=0.1,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    #Loading the best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    print(f"Loading the best model from: {best_model_path}")
    shutil.copy(best_model_path, saved_model_sampled if TRAIN_SUBSAMPLE else saved_model_full)
else:
    best_model_path = saved_model_sampled if TRAIN_SUBSAMPLE else saved_model_full
    load_weights(model, best_model_path)
    best_model =  model
    print ("Skipping Training and loading the model from {best_model_path}")