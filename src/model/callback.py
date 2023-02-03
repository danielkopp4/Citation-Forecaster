from tensorflow.keras.callbacks import Callback
import pandas as pd
import plotly.express as px
from datetime import datetime
from types import FunctionType
import os

# save plot of history
# print loss and metrics

class TrainingCallback(Callback):
    def __init__(
            self, 
            log_dir: str, 
            model_name: str, 
            plot_freq: int = 1, 
            print_freq: int = 1, 
            checkpoint_freq: int = 1,
            checkpoint_fn: FunctionType = None
        ):

        self._log_path: str = os.path.join(log_dir, model_name)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        if not os.path.exists(self._log_path):
            os.mkdir(self._log_path)

        self._train_info = pd.DataFrame()
        self._plot_freq = plot_freq
        self._print_freq = print_freq
        self._checkpoint_freq = checkpoint_freq
        self._checkpoint_fn = checkpoint_fn


    def save_train_plots(self):
        assert len(self._train_info) > 0
        for plot_type in self._train_info.columns:
            if plot_type == 'epoch':
                continue
            
            fig = px.line(self._train_info, x='epoch', y=plot_type)
            fig.write_image(os.path.join(self._log_path, '{}.png'.format(plot_type)))

        self._train_info.to_csv(os.path.join(self._log_path, 'log_data.csv'), index=False)


    def get_datapoint(self, epoch, logs):
        return_value = dict(logs)
        return_value['epoch'] = [epoch for _ in range(len(logs.keys()))]
        return return_value
    

    def print_log(self, logs, epoch=None):
        if epoch != None:
            print(f'[Epoch: {epoch}] ', end='')

        for key in logs.keys():
            print(f'[{key}: {logs[key]:0.2f}] ', end='')

        now_time = datetime.now()
        print(f'[Time: {now_time.strftime("%H:%M:%S")}] [Day: {now_time.strftime("%m/%d/%Y")}]')


    def checkpoint(self, epoch):
        if self._checkpoint_fn != None:
            self._checkpoint_fn(epoch)


    def on_epoch_end(self, epoch, logs=None):
        self._train_info: pd.DataFrame = self._train_info.append(pd.DataFrame(self.get_datapoint(epoch, logs)))
        if epoch % self._print_freq == 0:
            self.print_log(logs, epoch)

        if epoch % self._plot_freq == 0:
            self.save_train_plots()

        if epoch % self._checkpoint_freq == 0:
            self.checkpoint(epoch)


    def on_train_end(self, logs=None):
        self.save_train_plots()


    def on_test_end(self, logs=None):
        self.print_log(logs)
