# -*- coding: utf-8 -*-
import os
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models", "tcn"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tcn_model.pth"
)

SCALER_X_PATH = os.path.join(
    MODEL_DIR,
    "scaler_X.pkl"
)

SCALER_Y_PATH = os.path.join(
    MODEL_DIR,
    "scaler_y.pkl"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.json"
)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# 3. LOAD CONFIGURATION
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)


FEATURES = config["input_features"]

CHANNELS = config["channels"]

KERNEL_SIZE = config["kernel_size"]

DILATIONS = config["dilations"]

DROPOUT = config["dropout"]

MAX_LEAD_STEPS = config["max_lead_steps"]

PADDING_VALUE = config["padding_value"]


# ============================================================
# 4. LOAD SCALERS
# ============================================================

with open(
    SCALER_X_PATH,
    "rb"
) as f:

    scaler_X = pickle.load(f)


with open(
    SCALER_Y_PATH,
    "rb"
) as f:

    scaler_y = pickle.load(f)


# ============================================================
# 5. TCN TEMPORAL BLOCK
# ============================================================

class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        k,
        d,
        dropout
    ):

        super().__init__()

        padding = (k - 1) * d

        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            k,
            padding=padding,
            dilation=d
        )

        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            k,
            padding=padding,
            dilation=d
        )

        self.conv3 = nn.Conv1d(
            out_ch,
            out_ch,
            k,
            padding=padding,
            dilation=d
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.downsample = (

            nn.Conv1d(
                in_ch,
                out_ch,
                1
            )

            if in_ch != out_ch
            else None
        )


    def forward(self, x):

        out = self.conv1(x)

        out = out[
            :,
            :,
            :-self.conv1.padding[0]
        ]

        out = self.relu(out)

        out = self.dropout(out)


        out = self.conv2(out)

        out = out[
            :,
            :,
            :-self.conv2.padding[0]
        ]

        out = self.relu(out)

        out = self.dropout(out)


        out = self.conv3(out)

        out = out[
            :,
            :,
            :-self.conv3.padding[0]
        ]

        out = self.relu(out)

        out = self.dropout(out)


        res = (
            x
            if self.downsample is None
            else self.downsample(x)
        )

        return out + res


# ============================================================
# 6. TCN MODEL
# ============================================================

class TCN(nn.Module):

    def __init__(
        self,
        num_inputs,
        channels,
        kernel,
        dilations,
        dropout
    ):

        super().__init__()

        layers = []

        for i, d in enumerate(dilations):

            in_ch = (
                num_inputs
                if i == 0
                else channels[i - 1]
            )

            layers.append(
                TemporalBlock(
                    in_ch,
                    channels[i],
                    kernel,
                    d,
                    dropout
                )
            )

        self.network = nn.Sequential(
            *layers
        )

        self.fc = nn.Linear(
            channels[-1],
            2
        )


    def forward(self, x):

        y = self.network(x)

        y = self.fc(
            y[:, :, -1]
        )

        return torch.sigmoid(y)


# ============================================================
# 7. LOAD TRAINED MODEL
# ============================================================

model = TCN(
    num_inputs=len(FEATURES),
    channels=CHANNELS,
    kernel=KERNEL_SIZE,
    dilations=DILATIONS,
    dropout=DROPOUT
).to(DEVICE)


state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict
)

model.eval()


# ============================================================
# 8. PREPARE TRACK
# ============================================================

def prepare_track(track_df):
    """
    Prepare the complete cyclone track for TCN inference.

    track_df must contain the same 14 variables used during
    model training.

    Rows must be ordered chronologically.
    """

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    missing = [
        col
        for col in FEATURES
        if col not in track_df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required features:\n"
            + "\n".join(missing)
        )


    # --------------------------------------------------------
    # Select exact feature order
    # --------------------------------------------------------

    X_raw = track_df[
        FEATURES
    ].values.astype(float)


    # --------------------------------------------------------
    # Apply training scaler
    # --------------------------------------------------------

    X_scaled = scaler_X.transform(
        X_raw
    )


    # --------------------------------------------------------
    # Convert to tensor
    #
    # Original training shape:
    #
    # samples × features × time
    # --------------------------------------------------------

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Original training code uses left padding.
    #
    # For a single prediction we do not need padding when
    # using the complete current track.
    #
    # However, the TCN architecture requires the sequence
    # itself to be supplied in feature × time format.
    # --------------------------------------------------------

    X_tensor = X_tensor.unsqueeze(0)

    X_tensor = X_tensor.permute(
        0,
        2,
        1
    )


    return X_tensor


# ============================================================
# 9. PREDICT LANDFALL
# ============================================================

def predict_landfall(track_df):
    """
    Predict landfall latitude and longitude.

    Parameters
    ----------
    track_df : pandas.DataFrame
        Complete cyclone track up to the current observation.

    Returns
    -------
    dict
        Prediction information.
    """

    if len(track_df) < 2:

        raise ValueError(
            "At least two cyclone observations "
            "are required for prediction."
        )


    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    X_tensor = prepare_track(
        track_df
    )


    X_tensor = X_tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction_scaled = model(
            X_tensor
        ).cpu().numpy()


    # --------------------------------------------------------
    # Convert normalized output back to
    # actual latitude / longitude
    # --------------------------------------------------------

    prediction = scaler_y.inverse_transform(
        prediction_scaled
    )


    pred_lat = float(
        prediction[0, 0]
    )

    pred_lon = float(
        prediction[0, 1]
    )


    # --------------------------------------------------------
    # Current cyclone position
    # --------------------------------------------------------

    current_lat = float(
        track_df.iloc[-1]["Lat"]
    )

    current_lon = float(
        track_df.iloc[-1]["Lon"]
    )


    return {

        "predicted_landfall_lat":
            pred_lat,

        "predicted_landfall_lon":
            pred_lon,

        "current_lat":
            current_lat,

        "current_lon":
            current_lon,

        "number_of_observations":
            len(track_df),

        "model_version":
            config.get(
                "model_version",
                "unknown"
            )

    }


# ============================================================
# 10. TEST FUNCTION
# ============================================================

if __name__ == "__main__":

    print("\nTCN prediction module loaded successfully.")

    print("Model version:",config.get("model_version", "unknown"))

    print("Number of input features: ",len(FEATURES))

    print("\nRequired features:")

    for i, feature in enumerate(
        FEATURES,
        start=1
    ):

        print(i,".",feature)

    print(
        "\nModel file:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nReady for app.py"
    )