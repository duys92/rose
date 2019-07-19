from PyQt4 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from time import sleep
import sys
import struct, serial
from keras.models import load_model
from MultiStepLSTM import MultiStepLTSM
from scipy.stats import multivariate_normal
from sklearn.preprocessing import scale
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from pyqtgraph.dockarea import *
import time
from sklearn.preprocessing import scale, MinMaxScaler
from VoiceRecognition import VoiceRecognition
import logging
import atexit

import warnings

warnings.filterwarnings("ignore")

np.random.seed(123)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.Session(config=config)
# set_session(sess)


class QtHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        record = self.format(record)
        if record: XStream.stdout().write('%s\n' % record)
        # originally: XStream.stdout().write("{}\n".format(record))


logger = logging.getLogger(__name__)
handler = QtHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class XStream(QtCore.QObject):
    _stdout = None
    _stderr = None
    messageWritten = QtCore.pyqtSignal(str)

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg):
        if (not self.signalsBlocked()):
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        if (not XStream._stdout):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if (not XStream._stderr):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr


class MyDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)

        self._console = QtGui.QTextBrowser(self)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._console)
        self.setLayout(layout)

        XStream.stdout().messageWritten.connect(self._console.insertPlainText)
        XStream.stderr().messageWritten.connect(self._console.insertPlainText)


class VoiceRecognitionThread(QtCore.QThread):

    def __init__(self):
        super(VoiceRecognitionThread, self).__init__()
        self.vr = VoiceRecognition(model_name="lm_model/output_graph.pbmm",
                                   alphabet="lm_model/alphabet.txt",
                                   trie="lm_model/trie",
                                   lm="lm_model/lm.binary")

        self.results = None

    def __del__(self):
        self.wait()

    def run(self):
        self.vr.listen_and_do()
        # pass


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def serial_listen(serial, size):
    while True:
        if serial.in_waiting != size:
            serial.flushInput()
            incoming_data = serial.read(size=size)
            return incoming_data


class Packet(object):
    cmd_code = 0x00
    packet = None

    def set(self, values):
        pass

    def _pack(self, p1, p2):
        return struct.pack(">BHH", self.cmd_code, p1, p2)

    def send(self, ser):
        pass


class DepthControl(Packet):
    cmd_code = 0x01
    depth = None

    def set(self, values):
        self.depth = values

    def send(self, ser):
        self.packet = self._pack(self.depth, 0)
        ser.write(self.packet)


class PerfControl(Packet):
    cmd_code = 0x02
    perfs = []

    def set(self, values):
        self.perfs = values

    def send(self, ser):
        self.packet = self._pack(self.perfs[0], self.perfs[1])
        ser.write(self.packet)


def send_cmd():
    global package
    p = package

    try:
        dc = DepthControl()
        dc.set(int(p.depth_control.text()))
        dc.send(p.serial1)
    except ValueError as err:
        print(err)


def send_shoot():
    global package
    p = package
    try:
        pc = PerfControl()
        perf1, perf2 = p.perf1_control.text(), p.perf2_control.text()
        pc.set([int(perf1), int(perf2)])
        pc.send(p.serial1)

    except ValueError as err:
        print(err)


initial_depth = -1
prev_depth = 0
depth_rate_ftmin = 0
prev_depth_control = 0


def update():
    global package, initial_depth, prev_depth, depth_rate_ftmin, my_thread, prev_depth_control
    p = package

    if p.i == len(p.raw_data):
        p.i = 0


    data = serial_listen(p.serial0, size=27)
    cur_depth, prev_depth_raw, cur_temp, depth_update_rate, desired_depth, ccl, automatic, shoot_now, perf_mode = struct.unpack('ffffffbbb', data)

    #TODO Fix this
    if(shoot_now == 1):
        print("SHOOTING!!!!")

    if int(cur_depth) == int(prev_depth) or int(cur_depth) > 20000 or int(cur_depth) < -20000:
        depth_rate = abs(cur_depth - prev_depth_raw)
        depth_rate_ftmin = depth_rate * 60 * (1000.0 / (depth_update_rate))

        return
    # print(perf_mode)
    if my_thread.isRunning():
        cur_dctrl = my_thread.vr.depth_control
        if my_thread.vr.depth_control is not None and automatic == 0:
            prev_depth_control = cur_dctrl
            val = int(cur_dctrl[0])
            val = val.to_bytes(2, byteorder='big', signed=False)
            p.serial1.write(val)

        elif automatic == 1 and (
                my_thread.vr.depth_control != prev_depth_control) and not my_thread.vr.depth_control is None:
            prev_depth_control = cur_dctrl
            val = int(cur_dctrl[0])
            val = val.to_bytes(2, byteorder='big', signed=False)
            p.serial1.write(val)

    if automatic == 1:
        p.auto_pic.setPixmap(QtGui.QPixmap("auto_mode.png"))

        time_left = ((desired_depth - cur_depth) / depth_rate_ftmin) * 60.
        delta = time.strftime("%H:%M:%S", time.gmtime(time_left))
        p.time_left_label.setText("ETA: {}".format(delta))
        p.auto_pic.show()
    else:
        p.auto_pic.setPixmap(QtGui.QPixmap(""))
        p.time_left_label.setText("")

    p.i = int(cur_depth)
    if initial_depth < 0:
        initial_depth = p.i

    p.i -= initial_depth

    p.depth_label.setText("Depth: " + str(round(cur_depth, 2)))
    p.temp_label.setText("Temp: " + str(round(cur_temp, 2)))
    p.rate_label.setText("ft/min: " + str(round(depth_rate_ftmin, 2)))

    if ccl > 10.0 or ccl < -10.0:
        ccl = p.raw_data[p.i - 1]

    if p.cntr >= 50:
        p.cntr = 0

    if p.cntr >= 47:
        gain = 0.05 * (depth_rate_ftmin) + 1
    else:
        gain = 1
        if np.random.randint(0, 1001) < 10:
            ccl *= 10

    # print(depth_rate_ftmin)
    p.raw_data[p.i] = ccl * gain + (np.random.randn() / 10.0)
    p.line_speed_raw_data[p.i] = depth_rate_ftmin

    # print(p.number_of_batches)
    if p.i % p.m.batch_size == 0 and p.i != 0 and p.i >= p.minimum_index:
        x, y = p.m._prepare_seq2seq_data(scale(p.raw_data[p.s:p.e]), look_back=p.m.look_back, look_ahead=p.m.look_ahead)

        p.actual[:] = x

        pred = p.m.predict(p.actual, batch_size=p.m.batch_size)
        offset = int(p.m.batch_size * 2)
        pred_istart = p.e + (0 * p.m.batch_size + offset)
        pred_iend = p.e + (1 * p.m.batch_size + offset)

        p.predicted[pred_istart: pred_iend] = pred

        p.s += p.m.batch_size
        p.e += p.m.batch_size

    if p.predicted[p.i] != 0:
        p.error[p.i] = abs(p.raw_data[p.i] - p.predicted[p.i])
        p.p_val[p.i] = multivariate_normal.logpdf(p.error[p.i], p.p_mean, p.p_cov)

    p.ccl_plot.setData(p.raw_data[p.j:p.i, 0])
    p.pred_plot.setData(p.predicted[p.j:p.i + p.m.batch_size, 0])
    # p.error_plot.setData(p.error[p.j:p.i, 0]-1.0)
    p.pvalue_plot.setData(p.p_val[p.j:p.i, 0] - 2)
    p.line_speed.setData(p.line_speed_raw_data[p.j:p.i, 0])

    prev_depth = int(cur_depth)
    if p.i >= p.window_buffer:
        p.j += 1
    p.i += 1
    p.cntr += 1

    app.processEvents()  ## force complete redraw for every plot
    sleep(0.001)


path = 'ccl-model.keras'
serial0 = serial.Serial("/dev/ttyACM0", baudrate=115200)
serial1 = serial.Serial(port='/dev/ttyUSB0', timeout=0.01, baudrate=115200)
m = load_model(path, custom_objects={"MultiStepLTSM": MultiStepLTSM})
m.load_all(path + '.h5')

p_val_list = list(m.p_values.keys())[0]
p_mean = m.p_values[p_val_list]['mean']
p_cov = m.p_values[p_val_list]['cov']

app = pg.mkQApp()

buffer_size = 128000
window_buffer = 3000

minimum_index = m.batch_size + m.look_back + m.look_ahead

win = QtGui.QMainWindow()
win.resize(3000, 800)
area = DockArea()
win.setCentralWidget(area)

label_dock = Dock("Control Center", size=(5, 5))
ccl_dock = Dock("CCL", size=(450, 600))
temp_dock = Dock("LS", size=(450, 600))

area.addDock(label_dock, 'left')
area.addDock(ccl_dock, 'right')
area.addDock(temp_dock, 'bottom', ccl_dock)

w1 = pg.LayoutWidget()
label = QtGui.QLabel(""" Downhole Sim vA """)
depth_label = QtGui.QLabel(""" Depth: -1""")
temp_label = QtGui.QLabel("""Temp: -1""")
rate_label = QtGui.QLabel("""Ft/Min: -1""")
auto_pic = QtGui.QLabel("""""")
time_left_label = QtGui.QLabel("""""")
rose_output = MyDialog()

depth_control_tb = QtGui.QLineEdit()

perf1_tb = QtGui.QLineEdit()
perf1_label = QtGui.QLabel("""Perf1:""")
perf2_tb = QtGui.QLineEdit()
perf2_label = QtGui.QLabel("""Perf2: """)
send_perf_btn = QtGui.QPushButton("Shoot")
send_perf_btn.clicked.connect(send_shoot)

send_cmd_btn = QtGui.QPushButton("Go To Depth")
send_cmd_btn.clicked.connect(send_cmd)

w1.addWidget(label, row=0, col=0)
w1.addWidget(depth_label, row=1, col=0)
w1.addWidget(rate_label, row=1, col=1)
w1.addWidget(temp_label, row=2, col=0)
w1.addWidget(depth_control_tb, row=3, col=0)
w1.addWidget(time_left_label, row=3, col=1)
w1.addWidget(send_cmd_btn, row=4, col=0)
w1.addWidget(auto_pic, row=4, col=1)

w1.addWidget(perf1_label, row=5, col=0)
w1.addWidget(perf1_tb, row=5, col=1)

w1.addWidget(perf2_label, row=6, col=0)
w1.addWidget(perf2_tb, row=6, col=1)
w1.addWidget(send_perf_btn, row=7, colspan=2)
w1.addWidget(rose_output, row=8, col=0, colspan=2)

label_dock.addWidget(w1)

p = pg.PlotWidget(
    labels={'bottom': ('Depth'), 'left': 'CCL'}, name="Plot1"
)
p.setXRange(0, 1000)
p.setYRange(-6, 6)

ccl_dock.addWidget(p)

p2 = pg.PlotWidget(
    labels={'bottom': 'Depth', 'left': 'Line Speed (ft/min)'}, name="Plot2"
)
p2.setXRange(0, 1000)
p2.setYRange(0, 400)
temp_dock.addWidget(p2)

p2.setXLink("Plot1")

win.show()

my_thread = VoiceRecognitionThread()
my_thread.start()

package = Map({
    'serial0': serial0,
    'serial1': serial1,
    'depth_control': depth_control_tb,
    'perf1_control': perf1_tb,
    'perf2_control': perf2_tb,
    'ccl_plot': p.plot(),  # ccl curve
    'pred_plot': p.plot(pen='r'),  # predicted curve
    'error_plot': p.plot(pen='g'),
    'pvalue_plot': p.plot(pen='b'),
    'line_speed': p2.plot(),
    'pgPlot': p,  # plot object
    # 'remote_plot': rplt,
    'm': m,  # keras model
    'cntr': 0,  # helper variable
    's': 0,  # helper variable
    'e': minimum_index,  # helper variable
    'actual': np.zeros((m.batch_size, m.look_back, m.look_ahead)),
    # placeholder that will be used to predict
    'predicted': np.zeros((buffer_size, 1), dtype=np.float32),  # placeholder of vars that will hold predicted values
    'error': np.zeros((buffer_size, 1), dtype=np.float32),  # placeholder of vars that will hold error values
    'p_val': np.zeros((buffer_size, 1), dtype=np.float32) * np.nan,  # placeholder of vars that will hold p-values
    'raw_data': np.zeros((buffer_size, 1), dtype=np.float32),  # dynamic array that will grow in m.batch_sizes
    'line_speed_raw_data': np.zeros((buffer_size, 1), dtype=np.float32) * np.nan,
    'i': 0,  # general counter that will count forever with every call to update
    'j': 0,
    'capacity': buffer_size,
    'minimum_index': minimum_index,
    'window_buffer': window_buffer,
    'max_buffer': buffer_size,
    'p_cov': p_cov,
    'p_mean': p_mean,
    'scaler': scale,
    'depth_label': depth_label,
    'temp_label': temp_label,
    'rate_label': rate_label,
    'time_left_label': time_left_label,
    'auto_pic': auto_pic,
    #'rose_output': rose_output,
    'prev_depth': 0,
})

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


def bulk_save():
    global package
    p = package
    # dump numpy arrays as session_files for later post-processing
    file_name = "session_data/"
    np.save(file_name + "raw_data", p.raw_data)


atexit.register(bulk_save)
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()
