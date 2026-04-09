# import numpy as np

# def infer(rx_waveform: np.complex64 = None):
#     if rx_waveform is None:
#         print("ready!", flush=True)
#     else:
#         print("shape", rx_waveform.shape, flush=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.signal import correlate
def preprocessing_firstbatch(rx_wave: np.complex64, tx_wave: np.complex64, batch_size: int = int(2560)):
    tx_len = len(tx_wave)
    rx_wave = rx_wave[:int(batch_size)]

    # ------------------------------
    # Step 1: 全局corr，定位第一个cir
    # ------------------------------
    search_len = 3 * tx_len
    corr = correlate(rx_wave, tx_wave, mode='valid')
    global_start = np.argmax(np.abs(corr[:search_len])) - 3
    if global_start < 0:
        return None

    cir_list = []
    idx = global_start
    while idx + tx_len <= len(corr):
        # 截取trunc_len
        trunc_len = 16 # tx_len
        cir = corr[idx:idx+trunc_len]

        # 频域补零再回时域做上采样
        cfr = np.fft.fftshift(np.fft.fft(cir, n=trunc_len))
        target_len = 64
        total_pad = target_len - trunc_len
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        cfr = np.pad(cfr, (pad_left, pad_right), mode="constant")
        cir = np.fft.ifft(np.fft.ifftshift(cfr))
        
        #归一化
        cir = cir / np.max(np.abs(cir))
        
        cir_list.append(cir)
        idx += tx_len
    return np.array(cir_list[5:-5])

def infer(rx_wave, model, tag_model="bw10M_100s_standard"):
    print("shape", rx_wave.shape, flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tx_wave = np.load(os.path.join(script_dir, "dataset", "wave_table_chirp.npy"))

    print("preprocessing", rx_wave.shape, flush=True)
    X = preprocessing_firstbatch(rx_wave, tx_wave)
    print("preprocessing finished", X.shape, flush=True)
    if X is None or len(X) == 0:
        raise ValueError("preprocessing failed: no CIRs found")
    X_test = np.stack([np.real(X), np.imag(X)], axis=-1).astype(np.float32)
    print("stack finished", X_test.shape, flush=True)
    num_classes = 2 if "class" not in tag_model else 4

    print("model finished", flush=True)
    y_pred = model.predict(X_test, batch_size=512)
    print("prediction finished", flush=True)
    if num_classes == 2:
        y_pred_bin = (y_pred.flatten() >= 0.5).astype(int)
    else:
        y_pred_class = np.argmax(y_pred, axis=1)
        y_pred_bin = (y_pred_class != 0).astype(int)
    print("infererence result:", y_pred_bin, flush=True)
    #return np.bincount(y_pred_bin).argmax()
    return

def load_model_main(tag_model="bw10M_100s_standard"):
    num_classes = 2 if "class" not in tag_model else 4
    loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    metric = 'accuracy'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model = tf.keras.models.load_model(os.path.join(script_dir, "results", f'MA_model_{tag_model}.h5'), compile=False)
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=[metric]
    )
    return model