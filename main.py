import tensorflow as tf
import numpy as np
import traceback
import sys
import cv2

from utils import Dataset, plot_image_misclass
from capsnet import CapsNet
from capsnet_mod import CapsNetMod

from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)


def msg_box(type, msg_text):
    msg = QMessageBox()
    if type == 'error':
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
    if type == 'warning':
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning")
    elif type == 'info':
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Info")

    msg.setText(msg_text)
    msg.exec_()


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('main.ui', self)

        self.threadpool = QThreadPool()

        self.progBar.setMaximum(1)
        self.pb_Test_load.clicked.connect(lambda: self.load_model('test'))
        self.pb_Test_test.clicked.connect(lambda: self.test_model())
        self.pb_Exp_loadModel.clicked.connect(lambda: self.load_model('exp'))
        self.pb_Exp_loadImg.clicked.connect(lambda: self.load_image())
        self.pb_Exp_run.clicked.connect(lambda: self.run_predict())
        self.hs_Exp_1.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_2.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_3.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_4.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_5.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_6.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_7.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_8.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_9.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_10.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_11.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_12.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_13.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_14.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_15.valueChanged.connect(lambda: self.run_predict())
        self.hs_Exp_16.valueChanged.connect(lambda: self.run_predict())
        self.dataset = None
        self.data = None
        self.model = None

    def load_model(self, mode):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, 'QFileDialog.getOpenFileName()', '', 'All Files (*)', options=options)

        if not filename:
            return

        self.progBar.setMaximum(0)

        def load():
            if mode == 'test':
                self.dataset = str(self.cb_Test_dataset.currentText())
                model_idx = int(self.cb_Test_model.currentIndex())
            elif mode == 'exp':
                self.dataset = str(self.cb_Exp_dataset.currentText())
                model_idx = int(self.cb_Exp_model.currentIndex())

            if mode != 'exp':
                self.data = Dataset(self.dataset)

            if model_idx == 0:
                self.model = CapsNet(self.dataset, mode=mode, r=3)
            elif model_idx == 1:
                self.model = CapsNetMod(self.dataset, mode=mode)

            return self.model.load_weight(0, filename)

        def output(ok):
            if not ok:
                self.model = None
                msg_box('warning', 'Load data + model failed!')
            else:
                msg_box('info', 'Load data + model successful!')
            self.progBar.setMaximum(1)

        worker = Worker(load)
        worker.signals.result.connect(output)
        self.threadpool.start(worker)

    def test_model(self):
        if not self.data or not self.model:
            msg_box('warning', 'Model not loaded!')
            return

        self.progBar.setMaximum(0)

        def test():
            nclass = int(self.cb_Test_class.currentIndex())

            if nclass > 0:
                nclass -= 1
                y = tf.argmax(self.data.y_test, axis=-1)
                c = tf.constant(
                    nclass, dtype=y.dtype, shape=y.shape)
                indx = tf.where(tf.equal(y, c))
                x_test = tf.gather_nd(self.data.x_test, indx)
                y_test = tf.gather_nd(self.data.y_test, indx)
                x_test_orig = tf.gather_nd(self.data.x_test_orig, indx)
            else:
                x_test = self.data.x_test
                y_test = self.data.y_test
                x_test_orig = self.data.x_test_orig

            acc, err = self.model.evaluate(x_test, y_test)
            y_pred, _, _, _ = self.model.predict(x_test)

            return x_test_orig, y_test, acc, err, y_pred

        def output(res):
            x_test_orig, y_test, acc, err, y_pred = res

            self.le_Test_acc.setText(f'{acc*100:.2f}%')
            self.le_Test_err.setText(f'{err*100:.2f}%')

            plot_image_misclass(
                x_test_orig, y_test, y_pred, self.data.class_names)
            self.progBar.setMaximum(1)

        worker = Worker(test)
        worker.signals.result.connect(output)
        self.threadpool.start(worker)

    def load_image(self):
        options = QFileDialog.Options()
        data_dir, _ = QFileDialog.getOpenFileName(
            self, 'QFileDialog.getOpenFileName()', '', 'All Files (*)', options=options)

        if not data_dir:
            return

        self.progBar.setMaximum(0)

        # Reload data
        def load():
            self.dataset = str(self.cb_Exp_dataset.currentText())
            self.data = Dataset(self.dataset, custom_dir=data_dir)

        def output():
            try:
                self.data.x_custom
            except:
                self.data = None
                msg_box('warning', 'Load data failed!')
            else:
                msg_box('info', 'Load data successful!')
            self.progBar.setMaximum(1)

        worker = Worker(load)
        worker.signals.result.connect(output)
        self.threadpool.start(worker)

        # Plot image to screen
        try:
            img = cv2.imread(data_dir)
            qformat = QImage.Format_Indexed8
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(img,
                         img.shape[1],
                         img.shape[0],
                         img.strides[0],
                         qformat)
            img = img.rgbSwapped()
            pixmap = QPixmap.fromImage(img)
            pixmap = pixmap.scaled(self.label_Exp_img.size(),
                                   QtCore.Qt.KeepAspectRatio)
            self.label_Exp_img.setPixmap(pixmap)
            self.label_Exp_img.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        except:
            pass

    def run_predict(self):
        if not self.data or not self.model:
            msg_box('warning', 'Model or Image not loaded!')
            return

        self.progBar.setMaximum(0)

        def predict():
            # Get noise
            noise = np.array([float(self.hs_Exp_1.value()) / 100.0,
                              float(self.hs_Exp_2.value()) / 100.0,
                              float(self.hs_Exp_3.value()) / 100.0,
                              float(self.hs_Exp_4.value()) / 100.0,
                              float(self.hs_Exp_5.value()) / 100.0,
                              float(self.hs_Exp_6.value()) / 100.0,
                              float(self.hs_Exp_7.value()) / 100.0,
                              float(self.hs_Exp_8.value()) / 100.0,
                              float(self.hs_Exp_9.value()) / 100.0,
                              float(self.hs_Exp_10.value()) / 100.0,
                              float(self.hs_Exp_11.value()) / 100.0,
                              float(self.hs_Exp_12.value()) / 100.0,
                              float(self.hs_Exp_13.value()) / 100.0,
                              float(self.hs_Exp_14.value()) / 100.0,
                              float(self.hs_Exp_15.value()) / 100.0,
                              float(self.hs_Exp_16.value()) / 100.0])
            nclass = self.model.conf[f'class_{self.dataset}']
            noise = np.tile(noise, nclass)
            noise = np.reshape(noise, [1, nclass, 16])

            _, label, acc, x_reconstruct = self.model.predict(
                [self.data.x_custom, noise])

            x_reconstruct = tf.cast(x_reconstruct * 255.0, dtype=tf.int8)
            if x_reconstruct.shape[-1] == 2:
                x_reconstruct = x_reconstruct[..., 0]

            # Plot result to screen
            try:
                self.label_Exp_class.setText(
                    f'Label: {self.data.class_names[label[0]]}')
                self.label_Exp_acc.setText(f'Acc: {acc[0]*100:.2f}%')
                img = tf.squeeze(x_reconstruct).numpy()

                qformat = QImage.Format_Indexed8
                if len(img.shape) == 3:
                    if img.shape[2] == 4:
                        qformat = QImage.Format_RGBA8888
                    else:
                        qformat = QImage.Format_RGB888
                img = QImage(img,
                             img.shape[1],
                             img.shape[0],
                             img.strides[0],
                             qformat)
                img = img.rgbSwapped()
                pixmap = QPixmap.fromImage(img)
                # pixmap.save('test.jpg')
                pixmap = pixmap.scaled(self.label_Exp_img.size(),
                                       QtCore.Qt.KeepAspectRatio)
                self.label_Exp_img2.setPixmap(pixmap)
                self.label_Exp_img2.setAlignment(
                    QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            except:
                pass

        def output():
            self.progBar.setMaximum(1)

        worker = Worker(predict)
        worker.signals.result.connect(output)
        self.threadpool.start(worker)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()
