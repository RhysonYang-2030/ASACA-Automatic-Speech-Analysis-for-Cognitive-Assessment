#!/usr/bin/env python
# -*- coding: utf-8 -*-
# speech analysis for cognitive assenment  ASACA
import sys
try:
    sys.argv.remove("gui")      # 删除第一次出现的 "gui"
except ValueError:
    pass                        # 没找到就跳过
import os
import shutil
import time
import json
import pandas as pd
import argparse
import ctypes
import numpy as np
from pathlib import Path
from PyQt5 import QtGui                              # for QPixmap
from pathlib import Path
from PyQt5 import QtCore
# PyQt5 相关导入
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QPlainTextEdit, QTableWidget, QTableWidgetItem,
                             QFileDialog, QGroupBox, QFormLayout, QSplitter, QMessageBox, QProgressBar, QCheckBox)
from PyQt5.QtCore import QTimer, Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import (QAudioRecorder, QAudioEncoderSettings, QAudioProbe,
                                QAudioBuffer, QMultimedia, QAudioFormat)

# 用于绘制实时波形
import pyqtgraph as pg
from PyQt5.QtCore import QT_VERSION_STR

from asaca.inference import run_inference_and_seg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# ReportLab 用于生成 PDF 报告
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

import joblib
from asaca.cognition.cognition_inference import CognitionClassifier

# ======================= 加载配置文件 ===========================
def loadConfig():
    """Load ``config.json`` if present and fall back to defaults when needed."""

    config_path = "config.json"
    defaults = {
        "pretrained_processor": "xyang2/ASACA_ASR",
        "pretrained_model": "xyang2/ASACA_ASR",
        "output_dir": "output",
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {config_path}: {e}. Using defaults.")
            return defaults.copy()

        for k, v in defaults.items():
            if not cfg.get(k):
                cfg[k] = v
        return cfg

    return defaults.copy()


def mergeConfig(args, config):
    """
    将命令行参数覆盖到配置字典中（如果参数不为 None）。
    """
    for key in config.keys():
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            config[key] = arg_val
    return config




# ======================= 模型加载后台线程 ===========================
class ModelLoaderWorker(QThread):
    loaded = pyqtSignal(object, object)
    errorOccurred = pyqtSignal(str)

    def __init__(self, config, parent=None):
        super(ModelLoaderWorker, self).__init__(parent)
        self.config = config

    def run(self):
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            processor = Wav2Vec2Processor.from_pretrained(self.config["pretrained_processor"])
            model = Wav2Vec2ForCTC.from_pretrained(
                self.config["pretrained_model"],
                ctc_loss_reduction="mean",
                ctc_zero_infinity=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            self.loaded.emit(processor, model)
        except Exception as e:
            self.errorOccurred.emit(str(e))


# ======================= 推理后台线程 ===========================
class InferenceWorker(QThread):
    resultReady = pyqtSignal(str, dict, list)
    errorOccurred = pyqtSignal(str)

    def __init__(self, audioPath, model, processor, plot_output_dir="output", parent=None): # Define the plot output path
        super(InferenceWorker, self).__init__(parent)
        self.audioPath = audioPath
        self.model = model
        self.processor = processor
        self.plot_output_dir = plot_output_dir

    def run(self):
        try:
            annotated_text, global_features, dp_info = run_inference_and_seg(
                self.audioPath,
                self.model,
                self.processor,
                sr=16000,
                hop_length=512,
                decoder_method='beam_search',
                plot_output_dir=self.plot_output_dir  # 传入
            )
            self.resultReady.emit(annotated_text, global_features, dp_info)
        except Exception as e:
            self.errorOccurred.emit(str(e))


# ======================= 录音模块类 ===========================
class RecorderWidget(QWidget):
    """
    录音模块：提供录音控制，实时音量（dB）和实时波形显示，并在录音结束后保存文件并回调 onRecordingSaved。
    """

    def __init__(self, parent=None):
        super(RecorderWidget, self).__init__(parent)
        # 先定义波形环形缓冲区
        self.maxWaveLength = 960000
        self.waveData = np.array([], dtype=np.float32)

        self.initUI()
        self.audioRecorder = QAudioRecorder()
        self.setupRecorder()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTime)
        self.audioProbe = QAudioProbe()
        self.audioProbe.setSource(self.audioRecorder)
        self.audioProbe.audioBufferProbed.connect(self.processBuffer)

        self.startTime = None
        self.recordDuration = 0
        self.currentVolume = 0.0
        # 每个采样点对应的秒数(16kHz)
        self.dt = 16000.0

    def initUI(self):
        mainLayout = QVBoxLayout()

        self.statusLabel = QLabel("Status: Idle")
        self.timeLabel = QLabel("Time: 00:00")
        self.volumeLabel = QLabel("Volume: 0 dB")

        # 实时波形绘制
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.setYRange(-1.0, 1.0)
        self.plotWidget.setLabel("left", "Amplitude (normalized)")
        self.plotWidget.setLabel("bottom", "Time (s)")
        self.plotWidget.setXRange(0, self.maxWaveLength / 16000.0, padding=0)
        self.plotWidget.showGrid(x=True, y=True)
        self.curve = self.plotWidget.plot(pen='g')

        self.startButton = QPushButton("Start Recording")
        self.pauseButton = QPushButton("Pause")
        self.resumeButton = QPushButton("Resume")
        self.stopButton = QPushButton("Stop")
        self.pauseButton.setEnabled(False)
        self.resumeButton.setEnabled(False)
        self.stopButton.setEnabled(False)

        self.startButton.clicked.connect(self.startRecording)
        self.pauseButton.clicked.connect(self.pauseRecording)
        self.resumeButton.clicked.connect(self.resumeRecording)
        self.stopButton.clicked.connect(self.stopRecording)

        btnLayout = QHBoxLayout()
        btnLayout.addWidget(self.startButton)
        btnLayout.addWidget(self.pauseButton)
        btnLayout.addWidget(self.resumeButton)
        btnLayout.addWidget(self.stopButton)

        mainLayout.addWidget(self.statusLabel)
        mainLayout.addWidget(self.timeLabel)
        mainLayout.addWidget(self.volumeLabel)
        mainLayout.addWidget(self.plotWidget)
        mainLayout.addLayout(btnLayout)
        self.setLayout(mainLayout)

    def setupRecorder(self):
        settings = QAudioEncoderSettings()
        settings.setCodec("audio/pcm")
        settings.setQuality(QMultimedia.HighQuality)
        settings.setChannelCount(1)
        settings.setSampleRate(16000)
        self.audioRecorder.setEncodingSettings(settings)
        self.audioRecorder.setContainerFormat("wav")
        self.tempOutput = os.path.join(os.getcwd(), "temp_recording.wav")
        self.audioRecorder.setOutputLocation(QUrl.fromLocalFile(self.tempOutput))

        # 1) 选一个可用输入
        inputs = self.audioRecorder.audioInputs()
        if inputs:
            self.audioRecorder.setAudioInput(inputs[0])

        # 2) 调试信号（兼容 Qt5 / Qt6）
        if hasattr(self.audioRecorder, "errorOccurred"):  # Qt 6+
            self.audioRecorder.errorOccurred.connect(
                lambda err, s: print("Recorder error:", s)
            )
        else:  # Qt 5.x
            self.audioRecorder.error.connect(
                lambda err: print("Recorder error:", self.audioRecorder.errorString())
            )

        self.audioRecorder.stateChanged.connect(
            lambda st: print("Recorder state:", st)
        )

    def startRecording(self):
        self.waveData = np.array([], dtype=np.float32)
        self.audioRecorder.record()
        self.statusLabel.setText("Status: Recording")
        self.startTime = time.time()
        self.recordDuration = 0
        self.timer.start(1000)
        self.startButton.setEnabled(False)
        self.pauseButton.setEnabled(True)
        self.stopButton.setEnabled(True)
        self.resumeButton.setEnabled(False)

    def pauseRecording(self):
        self.audioRecorder.pause()
        self.statusLabel.setText("Status: Paused")
        self.timer.stop()
        self.pauseButton.setEnabled(False)
        self.resumeButton.setEnabled(True)

    def resumeRecording(self):
        self.audioRecorder.record()
        self.statusLabel.setText("Status: Recording")
        self.timer.start(1000)
        self.pauseButton.setEnabled(True)
        self.resumeButton.setEnabled(False)

    def stopRecording(self):
        self.audioRecorder.stop()
        self.timer.stop()
        self.statusLabel.setText("Status: Stopped")
        self.startButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.resumeButton.setEnabled(False)
        self.stopButton.setEnabled(False)
        savePath, _ = QFileDialog.getSaveFileName(self, "Save Recorded Audio", "", "WAV Files (*.wav)")
        if savePath:
            try:
                shutil.move(self.tempOutput, savePath)
                QMessageBox.information(self, "Recording Saved", f"Recording saved to:\n{savePath}")
                if hasattr(self, "onRecordingSaved"):
                    self.onRecordingSaved(savePath)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save file: {e}")
        else:
            if os.path.exists(self.tempOutput):
                os.remove(self.tempOutput)

    def updateTime(self):
        self.recordDuration += 1
        minutes = self.recordDuration // 60
        seconds = self.recordDuration % 60
        self.timeLabel.setText(f"Time: {minutes:02d}:{seconds:02d}")

    def processBuffer(self, buffer: QAudioBuffer):
        try:
            fmt = buffer.format()
            sampleType = fmt.sampleType()
            sampleSize = fmt.sampleSize()
            byteCount = buffer.byteCount()
            ptr = int(buffer.constData())
            sampleBytes = sampleSize // 8
            count = byteCount // sampleBytes

            if sampleType == QAudioFormat.SignedInt and sampleSize == 16:
                ctype = ctypes.c_short
                np_dtype = np.int16
            elif sampleType == QAudioFormat.UnSignedInt and sampleSize == 16:
                ctype = ctypes.c_ushort
                np_dtype = np.uint16
            elif sampleType == QAudioFormat.Float:
                ctype = ctypes.c_float
                np_dtype = np.float32
            else:
                self.volumeLabel.setText("Volume: Unsupported format")
                return

            array_type = ctype * count
            c_array = array_type.from_address(ptr)
            samples = np.array(c_array, dtype=np_dtype)
            if samples.size == 0:
                return

            wave_float = samples.astype(np.float32)
            if sampleType in (QAudioFormat.SignedInt, QAudioFormat.UnSignedInt):
                if sampleType == QAudioFormat.UnSignedInt:
                    wave_float = wave_float - 32768.0
                wave_float /= 32768.0

            # 计算音量 dB
            rms = np.sqrt(np.mean(wave_float ** 2))
            db = 20 * np.log10(rms) if rms > 1e-9 else -999.0
            if db < -100:
                self.volumeLabel.setText("Volume: -∞ dB")
            else:
                self.volumeLabel.setText(f"Volume: {db:.1f} dB")

            # 环形缓冲更新
            self.waveData = np.concatenate([self.waveData, wave_float])
            if len(self.waveData) > self.maxWaveLength:
                self.waveData = self.waveData[-self.maxWaveLength:]

            # 构造 x 数组（ms）
            x = np.arange(len(self.waveData), dtype=np.float32) / 16000.0
            self.curve.setData(x, self.waveData)

        except Exception as e:
            self.volumeLabel.setText("Volume: Error")
            print("Error in processBuffer:", e)


# ======================= 推理结果展示模块 ===========================
class InferenceResultWidget(QWidget):
    """
    展示：转录文本 + 全局特征表 + 风险标签 + 波形/音高图 + 雷达图，并支持 PDF 导出
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._buildUI()
        self.plotImagePath = ""
        self.shapPlotPath = None

        # ---------- UI ----------
    def _buildUI(self):
        main = QHBoxLayout(self)                         # 整体左右分栏

        # ===== 左侧：文本 & 全局特征 =====
        leftBox = QVBoxLayout()

        self.transcriptionEdit = QPlainTextEdit(readOnly=True)
        self.transcriptionEdit.setPlaceholderText("Transcription will appear here…")

        # ---- 全局特征表 ----
        self.featureTbl = QTableWidget()
        self.featureTbl.verticalHeader().setVisible(False)
        self.featureTbl.setColumnCount(2)
        self.featureTbl.setHorizontalHeaderLabels(["Metric", "Value"])
        self.featureTbl.horizontalHeader().setStretchLastSection(True)
        self.featureTbl.setFixedHeight(240)

        # ---- 风险标签 ----
        self.riskLabel = QLabel("—")
        self.riskLabel.setAlignment(Qt.AlignCenter)
        self.riskLabel.setStyleSheet("border:1px solid #888;padding:4px;font-weight:bold;")

        leftBox.addWidget(QLabel("Transcription"))
        leftBox.addWidget(self.transcriptionEdit, 2)
        leftBox.addWidget(QLabel("Global Features"))
        leftBox.addWidget(self.featureTbl)
        leftBox.addWidget(QLabel("Risk Flag"))
        leftBox.addWidget(self.riskLabel)

        # ---- Cognition classifier toggle ----
        self.cogCheck = QCheckBox("Run Cognition Classifier")
        self.shapCheck = QCheckBox("Show SHAP plot")
        self.cogCheck.setChecked(False)
        self.shapCheck.setChecked(False)
        leftBox.addWidget(self.cogCheck)
        leftBox.addWidget(self.shapCheck)


        # ===== 右侧：波形图 + 雷达 =====
        rightBox = QVBoxLayout()

        self.plotLabel = QLabel(alignment=Qt.AlignCenter)
        self.plotLabel.setMinimumHeight(280)
        self.plotLabel.setText("Inference Plot")

        # Matplotlib Radar
        self.radarFig = Figure(figsize=(3, 3), tight_layout=True)
        self.radarCanvas = FigureCanvas(self.radarFig)

        rightBox.addWidget(self.plotLabel, 3)
        rightBox.addWidget(self.radarCanvas, 2)
        # ---------- SHAP 图占位 ----------
        self.shapImg = QLabel(alignment=Qt.AlignCenter)
        self.shapImg.setVisible(False)  # 默认隐藏，勾选后再显示
        self.shapImg.setMaximumHeight(260)
        self.shapImg.setScaledContents(True)
        rightBox.addWidget(self.shapImg, 2)
        # ---------------------------------

        # ---- 导出 PDF 按钮 ----
        self.exportBtn = QPushButton("Export PDF Report")
        self.exportBtn.clicked.connect(self.exportReport)

        # ---- Assemble ----
        main.addLayout(leftBox, 3)
        main.addLayout(rightBox, 2)
        main.addWidget(self.exportBtn, alignment=Qt.AlignBottom)

    # ---------- 更新结果 ----------
    def updateResults(self, transcription: str, global_features: dict, plotImagePath: str):
        # 1. 文本
        self.transcriptionEdit.setPlainText(transcription)

        # 2. 全局特征（只显示标量）
        show_map = {
            "task_duration"        : "Task Duration (s)",
            "syllable_count"       : "Syllable Count",
            "speech_rate"          : "Speech Rate (/s)",
            "articulation_rate"    : "Articulation Rate (/s)",
            "pause_count"          : "Pause Count",
            "total_pause_duration" : "Total Pause (s)",
            "mean_pause_duration"  : "Mean Pause (s)",
            "pause_ratio"          : "Pause Ratio",
            "disfluency_count"     : "Disfluency Count",
        }

        rows = [(show_map[k], global_features.get(k, "—")) for k in show_map]
        self.featureTbl.setRowCount(len(rows))
        for i, (metric, val) in enumerate(rows):
            self.featureTbl.setItem(i, 0, QTableWidgetItem(metric))
            self.featureTbl.setItem(i, 1, QTableWidgetItem(str(val)))

        # 3. 风险标签
        if global_features.get("manual_review", False):
            self.riskLabel.setText("HIGH RISK – REVIEW NEEDED")
            self.riskLabel.setStyleSheet("background:#c62828;color:#fff;font-weight:bold;"
                                          "border-radius:6px;padding:4px;")
        else:
            self.riskLabel.setText("Low Risk ✔")
            self.riskLabel.setStyleSheet("background:#2e7d32;color:#fff;font-weight:bold;"
                                          "border-radius:6px;padding:4px;")

        # 4. Plot 图像
        self.plotImagePath = global_features.get("plot_path", plotImagePath)
        if os.path.exists(self.plotImagePath):
            pix = QPixmap(self.plotImagePath)
            self.plotLabel.setPixmap(pix.scaled(self.plotLabel.size(),
                                                Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation))
        else:
            self.plotLabel.setText(f"Plot not found:\n{self.plotImagePath}")

        # 5. Radar 图
        self._drawRadar(global_features)

    # ---------- Radar ----------
    def _drawRadar(self, feats: dict):
        metrics = ["speech_rate", "articulation_rate", "pause_ratio",
                   "mean_pause_duration", "disfluency_count"]
        labels  = ["SpeechRate", "ArticRate", "PauseRatio",
                   "MeanPause", "Disflu"]
        values = [float(feats.get(m, 0)) for m in metrics]
        # 简单归一化到 0-1（避免零数据）
        max_ref = [6, 6, 1, 1, 30]                 # 经验阈值
        data = [min(v / r, 1.0) for v, r in zip(values, max_ref)]
        data += data[:1]                            # 闭合

        angles = np.linspace(0, 2 * np.pi, len(labels) + 1)
        self.radarFig.clear()
        ax = self.radarFig.add_subplot(111, polar=True)
        ax.plot(angles, data, linewidth=2)
        ax.fill(angles, data, alpha=.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels([])
        ax.grid(True)
        self.radarCanvas.draw()

    # ---------- 重载 resize，以保持 plotLabel 缩放 ----------
    def resizeEvent(self, e):
        if self.plotLabel.pixmap():
            self.plotLabel.setPixmap(self.plotLabel.pixmap().scaled(
                self.plotLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(e)

    # ---------- PDF 导出 ----------
    def wrap_line_by_width(self, pdf_canvas, text: str,
                           max_width: float) -> list:
        words = text.strip().split()
        if not words: return [""]
        lines, cur = [], ""
        for w in words:
            test = f"{cur} {w}".strip()
            if stringWidth(test, "Helvetica", 12) <= max_width:
                cur = test
            else:
                lines.append(cur); cur = w
        lines.append(cur)
        return lines

    def write_multiline_text(self, can, text, x, y, max_w, spacing=15):
        lines = text.split('\n')
        for ln in lines:
            for seg in self.wrap_line_by_width(can, ln, max_w):
                if y < 50:                          # 新页
                    can.showPage(); can.setFont("Helvetica", 12)
                    y = letter[1] - 50
                can.drawString(x, y, seg); y -= spacing
        return y

    def exportReport(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Export Report", "", "PDF (*.pdf)")
        if not fn: return
        try:
            c = canvas.Canvas(fn, pagesize=letter)
            w, h = letter
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(w / 2, h - 50, "ASACA Inference Report")

            y = h - 80; c.setFont("Helvetica", 12)
            c.drawString(50, y, "Transcription:"); y -= 20
            y = self.write_multiline_text(c, self.transcriptionEdit.toPlainText(),
                                          60, y, 500)

            y -= 15; c.drawString(50, y, "Global Features:"); y -= 20
            for r in range(self.featureTbl.rowCount()):
                line = f"{self.featureTbl.item(r,0).text()}: {self.featureTbl.item(r,1).text()}"
                if y < 50: c.showPage(); c.setFont("Helvetica", 12); y = h - 50
                c.drawString(60, y, line); y -= 15

            # Cognition result (if any)
            if hasattr(self, "cogCheck") and self.cogCheck.isChecked():
                pred = getattr(self.parent().parent(), "cognition_prediction", None)
                if pred:
                    if y < 50: c.showPage(); c.setFont("Helvetica", 12); y = h - 50
                    c.drawString(60, y, f"Cognition Prediction: {pred}"); y -= 15

            y -= 20; c.drawString(50, y, "Plot:"); y -= 15

            c.drawString(60, y, "The image below shows the audio waveform and pitch curve.")
            y -= 20
            img_h = 300
            if y < img_h + 50: c.showPage(); c.setFont("Helvetica", 12); y = h - 50
            if os.path.exists(self.plotImagePath):
                c.drawImage(self.plotImagePath, 50, y - img_h,
                            width=500, height=img_h, preserveAspectRatio=True)
            y -= img_h + 20
            if getattr(self, "shapPlotPath", None) and os.path.exists(self.shapPlotPath):
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawCentredString(w / 2, h - 50, "Feature Contribution (SHAP)")
                img_h = 350
                c.drawImage(self.shapPlotPath, 50, h - 80 - img_h,
                            width=500, height=img_h, preserveAspectRatio=True)
                c.setFont("Helvetica", 10)
                c.drawString(50, h - 80 - img_h - 20,
                             "Positive bars push toward the predicted class; negative bars push away.")
            # 保存 & 提示
            c.save()
            QMessageBox.information(self, "Export", f"PDF saved:\n{fn}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

# ──────────────────────────────────────────────────────────
# ② 详细数据表  DetailedDataWidget
# ──────────────────────────────────────────────────────────
class DetailedDataWidget(QWidget):
    """
    表格展示对齐/音节等详细信息，可导出 Excel
    """
    def __init__(self, parent=None):
        super().__init__(parent); self._buildUI()

    def _buildUI(self):
        lay = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemDoubleClicked.connect(lambda it: QApplication.clipboard().setText(it.text()))
        lay.addWidget(self.table)

        self.exportBtn = QPushButton("Export Detailed Excel")
        self.exportBtn.clicked.connect(self.exportData)
        lay.addWidget(self.exportBtn, alignment=Qt.AlignRight)

    # ----- 更新 -----
    def updateData(self, data: list):
        if not data:
            self.table.clear(); self.table.setRowCount(0); self.table.setColumnCount(0); return
        cols = list(data[0].keys())
        self.table.setColumnCount(len(cols)); self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(data))
        for r, row in enumerate(data):
            for c, k in enumerate(cols):
                self.table.setItem(r, c, QTableWidgetItem(str(row.get(k, ""))))
        self.table.resizeColumnsToContents()

    # ----- 导出 -----
    def exportData(self):
        if self.table.rowCount() == 0: return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Detailed Data", "", "Excel (*.xlsx)")
        if not fn: return
        try:
            df = pd.DataFrame([[self.table.item(r, c).text()
                                for c in range(self.table.columnCount())]
                               for r in range(self.table.rowCount())],
                              columns=[self.table.horizontalHeaderItem(i).text()
                                       for i in range(self.table.columnCount())])
            df.to_excel(fn, index=False, engine="openpyxl")
            QMessageBox.information(self, "Export", f"Saved to:\n{fn}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

# ──────────────────────────────────────────────────────────
# ③ 主窗口  MainWindow
# ──────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """
    录音、推理、详细结果 三页 + 拖放 / 打开音频支持 + 状态彩灯
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("ASACA – Speech Analysis")
        self.resize(1280, 860)
        self.setAcceptDrops(True)                 # 拖放音频
        self._buildUI()
        self.loadInferenceModel()

    # ---------- UI ----------
    def _buildUI(self):
        self.tabs = QTabWidget()

        # RecorderWidget
        self.recordingTab = RecorderWidget(); self.recordingTab.onRecordingSaved = self.runInference
        self.inferenceTab = InferenceResultWidget()
        self.detailTab    = DetailedDataWidget()

        self.tabs.addTab(self.recordingTab, "Recording")
        self.tabs.addTab(self.inferenceTab, "Inference Result")
        self.tabs.addTab(self.detailTab, "Detailed Data")

        self.openFileButton = QPushButton("Open Audio File")
        self.openFileButton.clicked.connect(self.onOpenAudioFile)

        cen = QWidget(); lay = QVBoxLayout(cen)
        lay.addWidget(self.openFileButton, 0, Qt.AlignLeft)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(cen)

        # 状态栏
        self.progressBar = QProgressBar(maximum=0, minimum=0, visible=False)
        self.statusLed   = QLabel("●"); self.statusLed.setStyleSheet("color:#2e7d32;font-size:16px;")

        # NEW —— Praat badge
        self.praatBadge = QLabel("Praat ✖")
        self.praatBadge.setStyleSheet("color:#9e9e9e;font-weight:bold;")

        # keep the order: coloured dot · badge · spinner
        sb = self.statusBar()
        sb.addPermanentWidget(self.statusLed)
        sb.addPermanentWidget(self.praatBadge)
        sb.addPermanentWidget(self.progressBar)



    def _setStatus(self, msg, color):
        self.statusBar().showMessage(msg)
        self.statusLed.setStyleSheet(f"color:{color};font-size:16px;")

    def _setPraatBadge(self, ok: bool):
        """
        Turn the badge green (✓) when Praat numbers are present,
        grey (✖) when the pipeline is disabled or failed.
        """
        if ok:
            self.praatBadge.setText("Praat ✓")
            self.praatBadge.setStyleSheet("color:#2e7d32;font-weight:bold;")
        else:
            self.praatBadge.setText("Praat ✖")
            self.praatBadge.setStyleSheet("color:#9e9e9e;font-weight:bold;")

    # ---------- 拖放 ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if url.isLocalFile():
                self.runInference(url.toLocalFile()); break

    # ---------- 打开文件 ----------
    def onOpenAudioFile(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select Audio",
                                            "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")
        if fp: self.runInference(fp)

    # ---------- 模型加载 ----------
    def loadInferenceModel(self):
        self._setStatus("Loading model…", "#f9a825"); self.progressBar.setVisible(True)
        self.modelLoader = ModelLoaderWorker(self.cfg)
        self.modelLoader.loaded.connect(self._onModelLoaded)
        self.modelLoader.errorOccurred .connect(self._onModelError)
        self.modelLoader.start()

    def _onModelLoaded(self, processor, model):
        import torch
        self.processor, self.model = processor, model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.progressBar.setVisible(False)
        self._setStatus("Model ready", "#2e7d32")
        self._loadCognitionClassifier()
    # ---------- Cognition model ----------
    def _loadCognitionClassifier(self):
        cfg = self.cfg.get("cognition", {})       # add this block to config.json later
        try:
            self.cogCls = CognitionClassifier(
                model_pkl      = Path(cfg.get("model_pkl",  "cognition_training/classifier.pkl")),
                dict_dir       = Path(cfg.get("dict_dir",   "dicts")),
                processor_path = cfg.get("pretrained_processor", self.cfg["pretrained_processor"]),
                asr_model_path = cfg.get("pretrained_model",     self.cfg["pretrained_model"]),
                device         = self.device,
                feature_xlsx=Path("cognition_training/TUH_FV_Improved.xlsx")
            )
            print("[Cognition] classifier ready")
        except Exception as e:
            print("[Cognition] load failed:", e)
            self.cogCls = None

    def _onModelError(self, err):
        QMessageBox.critical(self, "Model Error", err)
        self.progressBar.setVisible(False); self._setStatus("Model error", "#c62828")

    # ---------- 推理 ----------
    def runInference(self, audioPath):
        self.currentAudioPath = audioPath
        if not hasattr(self, "model"):
            QMessageBox.warning(self, "Warning", "Model not ready"); return
        out_dir = self.cfg.get("output_dir", "output"); os.makedirs(out_dir, exist_ok=True)
        self._setPraatBadge(False)  # NEW line
        self._setStatus("Processing…", "#f9a825"); self.progressBar.setVisible(True)

        self.inferWorker = InferenceWorker(audioPath, self.model, self.processor, out_dir)
        self.inferWorker.resultReady.connect(self._onInferOk)
        self.inferWorker.errorOccurred .connect(self._onInferErr)
        self.inferWorker.start()

    def _onInferOk(self, annotated_text, global_features, dp_info):
        plot_path = global_features.get("plot_path", "")
        self.inferenceTab.updateResults(annotated_text, global_features, plot_path)
        self.detailTab.updateData(dp_info)

        # ---- Cognition prediction (if toggled) ----
        pred = None
        if hasattr(self, "cogCls"):
            if self.inferenceTab.cogCheck.isChecked():  # user wants label
                pred = self.cogCls.predict_label(
                    transcription=annotated_text,
                    global_feats=global_features,
                )
                self.inferenceTab.riskLabel.setText(f"Cognition: {pred}")
                global_features["cognition_prediction"] = pred
            else:
                # still get class name for SHAP, but don’t overwrite risk label
                pred = self.cogCls.predict_label(
                    transcription=annotated_text,
                    global_feats=global_features,
                )

        self.tabs.setCurrentWidget(self.inferenceTab)
        self.progressBar.setVisible(False)
        self._setPraatBadge(bool(global_features.get("praat_success")))

        self._setStatus("Done", "#2e7d32")
        # ---------------- SHAP on demand ----------------
        if self.inferenceTab.shapCheck.isChecked() and hasattr(self, "cogCls"):
            shap_png = Path("tmp_shap.png")  # temporary file
            try:
                self.cogCls.explain(
                    class_name=pred,
                    save_png=shap_png,
                    transcription=annotated_text,
                    global_feats=global_features,
                )
                pix = QtGui.QPixmap(str(shap_png))
                self.inferenceTab.shapImg.setPixmap(
                    pix.scaled(
                        self.inferenceTab.shapImg.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
                self.inferenceTab.shapImg.setVisible(True)  # ← make visible
                self.inferenceTab.shapPlotPath = str(shap_png)  # ← PDF relies on it
                global_features["shap_plot"] = str(shap_png)  # for PDF
            except Exception as e:
                print("[SHAP] explanation failed:", e)
                self.inferenceTab.shapImg.setVisible(False)
                self.inferenceTab.shapPlotPath = None
        else:
            self.inferenceTab.shapImg.setVisible(False)
            self.inferenceTab.shapPlotPath = None
            pass



    def _onInferErr(self, err):
        QMessageBox.critical(self, "Inference Error", err)
        self.progressBar.setVisible(False)
        self._setPraatBadge(False)  # NEW
        self._setStatus("Error", "#c62828")

def main():
    parser = argparse.ArgumentParser(description="Clinical Speech Analysis Interface (Inference)")
    parser.add_argument("--pretrained_processor", type=str, help="Path to pretrained processor")
    parser.add_argument("--pretrained_model", type=str, help="Path to pretrained model")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()

    config = loadConfig()
    config = mergeConfig(args, config)

    app = QApplication(sys.argv)
    qss = """
        QWidget {
            font-family: "Segoe UI", sans-serif;
            font-size: 12pt;
            color: #000000;
            background-color: #FFFFFF;
        }
        QMainWindow {
            background-color: #FFFFFF;
        }
        QPushButton {
            background-color: #003366;
            color: #FFFFFF;
            border-radius: 8px;
            padding: 6px;
        }
        QPushButton:hover {
            background-color: #005599;
        }
        QPlainTextEdit, QTableWidget {
            border: 1px solid #003366;
            border-radius: 4px;
        }
        QGroupBox {
            border: 1px solid #003366;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
            color: #003366;
            font-size: 14pt;
        }
        QStatusBar {
            background-color: #E0E0E0;
        }
    """
    app.setStyleSheet(qss)
    mainWin = MainWindow(config)
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
