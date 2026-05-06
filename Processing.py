import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import segyio
import time
from model3D import *
from model2D import *

# -------------------------- Global Configuration --------------------------
UNET_DOWNSAMPLE_FACTOR = 16

# -------------------------- Basic Utility Functions --------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pad_to_valid_size(data, factor):
    """Pad data to dimensions divisible by factor for UNet compatibility"""
    pad_dims = []
    for dim in data.shape:
        remainder = dim % factor
        if remainder == 0:
            pad_dims.append((0, 0))
        else:
            pad_len = factor - remainder
            pad_before = pad_len // 2
            pad_after = pad_len - pad_before
            pad_dims.append((pad_before, pad_after))
    padded_data = np.pad(data, pad_dims, mode='reflect')
    return padded_data, pad_dims


def crop_back_to_original(padded_data, pad_dims):
    """Crop padded data back to original dimensions"""
    slices = []
    for pad_before, pad_after in pad_dims:
        start = pad_before
        end = None if pad_after == 0 else -pad_after
        slices.append(slice(start, end))
    return padded_data[tuple(slices)]


def split_block_3d(data, block_size, overlap):
    """Split 3D data into overlapping blocks for GPU memory efficiency"""
    blocks = []
    step = [int(s * (1 - overlap)) for s in block_size]
    for i in range(0, data.shape[0], step[0]):
        for j in range(0, data.shape[1], step[1]):
            for k in range(0, data.shape[2], step[2]):
                block = data[i:min(i + block_size[0], data.shape[0]),
                        j:min(j + block_size[1], data.shape[1]),
                        k:min(k + block_size[2], data.shape[2])]
                blocks.append((block, i, j, k))
    return blocks


def create_weight_map(block_shape, overlap):
    """Create 3D weight map for smooth block fusion"""
    weights = np.ones(block_shape)
    overlap_size = [int(s * overlap) for s in block_shape]
    for i in range(overlap_size[0]):
        weights[i, :, :] *= (i + 1) / (overlap_size[0] + 1)
        weights[-i - 1, :, :] *= (i + 1) / (overlap_size[0] + 1)
    for j in range(overlap_size[1]):
        weights[:, j, :] *= (j + 1) / (overlap_size[1] + 1)
        weights[:, -j - 1, :] *= (j + 1) / (overlap_size[1] + 1)
    for k in range(overlap_size[2]):
        weights[:, :, k] *= (k + 1) / (overlap_size[2] + 1)
        weights[:, :, -k - 1] *= (k + 1) / (overlap_size[2] + 1)
    return weights


def combine_blocks_3d(blocks, data_shape, block_size, overlap):
    """Fuse 3D blocks with weighted averaging to remove boundary artifacts"""
    result = np.zeros(data_shape)
    count = np.zeros(data_shape)
    step = [int(s * (1 - overlap)) for s in block_size]

    for block, i, j, k in blocks:
        block_shape = block.shape
        weights = create_weight_map(block_shape, overlap)
        i_end = min(i + block_shape[0], data_shape[0])
        j_end = min(j + block_shape[1], data_shape[1])
        k_end = min(k + block_shape[2], data_shape[2])

        block_i_slice = slice(0, i_end - i)
        block_j_slice = slice(0, j_end - j)
        block_k_slice = slice(0, k_end - k)

        result[i:i_end, j:j_end, k:k_end] += block[block_i_slice, block_j_slice, block_k_slice] * weights[
            block_i_slice, block_j_slice, block_k_slice]
        count[i:i_end, j:j_end, k:k_end] += weights[block_i_slice, block_j_slice, block_k_slice]

    count[count == 0] = 1
    result /= count
    return result


# -------------------------- SEGY I/O Core Functions --------------------------
def read_segy_original_logic(file_path, line_total_num=None, trace_total_num=None, log_func=None):
    def log(msg):
        if log_func:
            log_func(msg)
        else:
            print(msg)

    log("Reading SEGY file...")
    with segyio.open(file_path, "r", ignore_geometry=True) as src:
        total_trace_count = len(src.trace)
        sample_num = src.samples.size
        log(f"   Total traces: {total_trace_count}")
        log(f"   Samples per trace: {sample_num}")

        if line_total_num is None or trace_total_num is None:
            log("   Auto-detecting line and trace dimensions...")
            factors = []
            for i in range(1, int(np.sqrt(total_trace_count)) + 1):
                if total_trace_count % i == 0:
                    factors.append((i, total_trace_count // i))
            if factors:
                line_total_num, trace_total_num = factors[-1]
                log(f"   Detected: Lines={line_total_num}, Traces={trace_total_num}")
            else:
                line_total_num = 1
                trace_total_num = total_trace_count
                log(f"   Detected 2D data: Lines={line_total_num}, Traces={trace_total_num}")

        full_data = np.zeros((sample_num, trace_total_num, line_total_num), dtype=np.float32)
        for iline in range(line_total_num):
            start = iline * trace_total_num
            end = start + trace_total_num
            if end <= total_trace_count:
                full_data[:, :, iline] = np.array([src.trace[trace] for trace in range(start, end)]).T

        log(f"✅ Read completed, data shape: {full_data.shape} [sample, trace, line]")

        src_meta = {
            'format': src.format,
            'text_header': src.text[0],
            'bin_header': {},
            'tracecount': total_trace_count,
            'sample_num_original': sample_num,
            'samples_array': src.samples,
            'dt': src.bin[segyio.BinField.Interval] if segyio.BinField.Interval in src.bin else 4000,
            'line_total_num': line_total_num,
            'trace_total_num': trace_total_num
        }

        fields_to_save = [
            segyio.BinField.JobID, segyio.BinField.LineNumber, segyio.BinField.ReelNumber,
            segyio.BinField.Traces, segyio.BinField.AuxTraces, segyio.BinField.Interval,
            segyio.BinField.IntervalOriginal, segyio.BinField.SamplesOriginal, segyio.BinField.Format,
            segyio.BinField.EnsembleFold, segyio.BinField.SortingCode, segyio.BinField.VerticalSum,
            segyio.BinField.MeasurementSystem, segyio.BinField.Samples
        ]
        for field in fields_to_save:
            try:
                src_meta['bin_header'][field] = src.bin[field]
            except:
                pass

        return full_data, src_meta


def write_segy_original_logic(file_path, data_to_write, src_meta, log_func=None):
    def log(msg):
        if log_func:
            log_func(msg)
        else:
            print(msg)

    start_time = time.time()
    log("Writing SEGY file...")

    if len(data_to_write.shape) != 3:
        raise ValueError(f"Input must be 3D [sample, trace, line], current shape: {data_to_write.shape}")

    sample_num_out, trace_num_out, line_num_out = data_to_write.shape
    line_total_num = src_meta['line_total_num']
    trace_total_num = src_meta['trace_total_num']
    total_trace_count = src_meta['tracecount']

    spec = segyio.spec()
    spec.format = src_meta['format']
    spec.tracecount = total_trace_count
    if sample_num_out == src_meta['sample_num_original']:
        spec.samples = src_meta['samples_array']
    else:
        dt_ms = src_meta['dt'] / 1000.0
        spec.samples = np.arange(0, sample_num_out * dt_ms, dt_ms)

    with segyio.create(file_path, spec) as dst:
        dst.text[0] = src_meta['text_header']

        bin_header = src_meta['bin_header']
        fields_to_write = [
            segyio.BinField.JobID, segyio.BinField.LineNumber, segyio.BinField.ReelNumber,
            segyio.BinField.Traces, segyio.BinField.AuxTraces, segyio.BinField.Interval,
            segyio.BinField.IntervalOriginal, segyio.BinField.SamplesOriginal, segyio.BinField.Format,
            segyio.BinField.EnsembleFold, segyio.BinField.SortingCode, segyio.BinField.VerticalSum,
            segyio.BinField.MeasurementSystem
        ]
        for field in fields_to_write:
            if field in bin_header:
                dst.bin[field] = bin_header[field]
        dst.bin[segyio.BinField.Samples] = sample_num_out

        log(f"   Writing {total_trace_count} traces in original order...")
        for iline in range(line_total_num):
            trace_start = iline * trace_total_num
            trace_end = trace_start + trace_total_num
            if trace_end > total_trace_count:
                break

            line_data = data_to_write[:, :, iline]
            line_data_t = line_data.T

            for local_trace_idx in range(trace_total_num):
                global_trace_idx = trace_start + local_trace_idx
                if global_trace_idx >= total_trace_count:
                    break
                dst.trace[global_trace_idx] = line_data_t[local_trace_idx, :].astype(np.float32)

    log(f"✅ Write completed, time elapsed: {time.time() - start_time:.2f}s")


# -------------------------- Two-stage 3D Fault Detection Pipeline --------------------------
def process_data(file_path, log_func, progress_func):
    try:
        block_size = (128, 128, 256)
        overlap = 0.5
        src_meta = None

        log_func("Loading 2D+3D models...")
        model1 = load_model("unet2D.hdf5", custom_objects={"loss1": loss1})
        model2 = load_model("unet3D.hdf5", custom_objects={"loss1": loss1})
        log_func("✅ Models loaded successfully")
        progress_func(2)

        log_func("=" * 50)
        ext = os.path.splitext(file_path)[1].lower()
        data = None

        if ext in ['.segy', '.sgy']:
            data, src_meta = read_segy_original_logic(
                file_path,
                line_total_num=None,
                trace_total_num=None,
                log_func=log_func
            )
            data = data.transpose(2, 1, 0)
            log_func(f"   Transposed to processing shape: {data.shape} [line, trace, sample]")
        else:
            log_func("Reading DAT binary file...")
            data = np.fromfile(file_path, dtype=np.single)
            size = data.size
            cube_size = round(size ** (1 / 3))
            if cube_size ** 3 == size:
                data = data.reshape(cube_size, cube_size, cube_size)
            else:
                data = data.reshape(1, -1, cube_size)
            log_func(f"✅ DAT read completed, data shape: {data.shape}")

        progress_func(5)
        original_shape = data.shape

        log_func("=" * 50)
        log_func("Preprocessing data for UNet...")
        data_padded, pad_dims = pad_to_valid_size(data, UNET_DOWNSAMPLE_FACTOR)
        log_func(f"   Original size: {data.shape}")
        log_func(f"   Padded size: {data_padded.shape}")
        progress_func(8)

        len1_pad, len2_pad, len3_pad = data_padded.shape

        log_func("=" * 50)
        log_func("Stage 1: 2D Line feature extraction...")
        dataup = np.zeros((len1_pad, len2_pad, len3_pad))
        datadown = np.zeros((len1_pad, len2_pad, len3_pad))
        data2 = np.zeros((len1_pad, len2_pad, len3_pad))
        data3 = np.zeros((len1_pad, len2_pad, len3_pad))

        for i in range(len1_pad):
            dataper = data_padded[i, :, :]
            slice_padded, slice_pad = pad_to_valid_size(dataper, UNET_DOWNSAMPLE_FACTOR)
            slice_padded = np.reshape(slice_padded, (1, *slice_padded.shape, 1))
            dataper_pred = model1.predict(slice_padded, verbose=0)
            pred_up = crop_back_to_original(dataper_pred[0, :, :, 0], slice_pad)
            pred_down = crop_back_to_original(dataper_pred[0, :, :, 1], slice_pad)

            dataup[i, :, :] = pred_up
            datadown[i, :, :] = pred_down
            data2[i, :, :] = pred_up + pred_down

            if i % max(1, len1_pad // 10) == 0:
                progress_func(8 + int(i / len1_pad * 25))
                log_func(f"   Line progress: {i}/{len1_pad}")

        log_func("Stage 1: 2D Trace feature extraction...")
        for j in range(len2_pad):
            dataper = data_padded[:, j, :]
            slice_padded, slice_pad = pad_to_valid_size(dataper, UNET_DOWNSAMPLE_FACTOR)
            slice_padded = np.reshape(slice_padded, (1, *slice_padded.shape, 1))
            dataper_pred = model1.predict(slice_padded, verbose=0)

            pred_up = crop_back_to_original(dataper_pred[0, :, :, 0], slice_pad)
            pred_down = crop_back_to_original(dataper_pred[0, :, :, 1], slice_pad)

            dataup[:, j, :] = pred_up
            datadown[:, j, :] = pred_down
            data3[:, j, :] = pred_up + pred_down

            if j % max(1, len2_pad // 10) == 0:
                progress_func(33 + int(j / len2_pad * 20))
                log_func(f"   Trace progress: {j}/{len2_pad}")

        log_func("=" * 50)
        log_func("Calculating multi-direction attention weights...")
        dataattention = np.maximum(data2, data3)
        data_processed = data_padded * tf.sigmoid(dataattention).numpy()
        data_processed = (data_processed - np.min(data_processed)) / (np.max(data_processed) - np.min(data_processed))
        progress_func(55)

        log_func("=" * 50)
        log_func("Stage 2: Splitting 3D data into blocks...")
        blocks = split_block_3d(data_processed, block_size, overlap)
        total_blocks = len(blocks)
        dataups, datadowns = [], []

        log_func(f"Starting 3D prediction with {total_blocks} blocks")
        for idx, (block, i, j, k) in enumerate(blocks):
            block_padded, block_pad = pad_to_valid_size(block, UNET_DOWNSAMPLE_FACTOR)
            block_input = np.reshape(block_padded, (1, *block_padded.shape))
            pred = model2.predict(block_input, verbose=0)
            pred_up = crop_back_to_original(pred[0, ..., 0], block_pad)
            pred_down = crop_back_to_original(pred[0, ..., 1], block_pad)

            dataups.append((pred_up, i, j, k))
            datadowns.append((pred_down, i, j, k))

            if idx % max(1, total_blocks // 10) == 0:
                progress_func(55 + int(idx / total_blocks * 35))
                log_func(f"   3D block progress: {idx + 1}/{total_blocks}")

        log_func("=" * 50)
        log_func("Fusing 3D block results...")
        res1_padded = combine_blocks_3d(dataups, data_padded.shape, block_size, overlap)
        res2_padded = combine_blocks_3d(datadowns, data_padded.shape, block_size, overlap)
        result_padded = res1_padded + res2_padded

        log_func("Cropping back to original dimensions...")
        result = crop_back_to_original(result_padded, pad_dims)
        progress_func(95)

        log_func("=" * 50)
        out_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        if ext in ['.segy', '.sgy']:
            log_func("Converting to SEGY storage format...")
            result_for_segy = result.transpose(2, 1, 0)
            out_name = f"{base_name}_fault_detection_result.sgy"
            out_path = os.path.join(out_dir, out_name)
            write_segy_original_logic(out_path, result_for_segy, src_meta, log_func)
        else:
            out_name = f"{base_name}_fault_detection_result.dat"
            out_path = os.path.join(out_dir, out_name)
            result.astype(np.float32).tofile(out_path)
            log_func(f"✅ DAT result saved successfully")

        progress_func(100)
        log_func("=" * 50)
        log_func("🎉 Two-stage fault detection completed successfully!")
        messagebox.showinfo("Completed", f"Result saved to:\n{out_path}")

    except Exception as e:
        log_func(f"❌ Error occurred: {str(e)}")
        import traceback
        log_func(traceback.format_exc())
        messagebox.showerror("Processing Error", str(e))


# -------------------------- GUI Application --------------------------
class FaultDetectApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Two-stage 3D Fault Detection V4.0 Stable")
        self.geometry("780x580")
        self.resizable(False, False)
        self.file_path = ""

        ttk.Label(self, text="Two-stage 3D Fault Detection System", font=("Microsoft YaHei", 16, "bold")).pack(pady=12)

        frame_file = ttk.Frame(self)
        frame_file.pack(pady=8, fill=tk.X, padx=30)
        ttk.Label(frame_file, text="Data File:", font=("Microsoft YaHei", 10)).grid(row=0, column=0, sticky="w")
        self.file_label = ttk.Label(frame_file, text="No file selected (Supports .segy/.sgy/.dat)", width=48, foreground="gray")
        self.file_label.grid(row=0, column=1, padx=10)
        ttk.Button(frame_file, text="Select File", command=self.select_file, width=12).grid(row=0, column=2)

        ttk.Label(self, text="💡 Feature: Block processing for large-volume data",
                  font=("Microsoft YaHei", 9), foreground="gray").pack(pady=2)

        self.progress = ttk.Progressbar(self, length=700, mode='determinate')
        self.progress.pack(pady=12)

        ttk.Label(self, text="Runtime Log:", font=("Microsoft YaHei", 10)).pack(anchor="w", padx=30)
        self.log_text = tk.Text(self, height=16, width=95)
        self.log_text.pack(pady=5, padx=30)
        self.log_text.config(state=tk.DISABLED)

        self.run_btn = ttk.Button(self, text="Start Detection", command=self.start_run, width=22)
        self.run_btn.pack(pady=15)

    def select_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("SEGY Data", "*.segy;*.sgy"), ("DAT Binary", "*.dat"), ("All Files", "*.*")]
        )
        if path:
            self.file_path = path
            self.file_label.config(text=os.path.basename(path), foreground="black")

    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_progress(self, val):
        self.progress["value"] = val

    def start_run(self):
        if not self.file_path:
            messagebox.showwarning("Warning", "Please select a data file first!")
            return
        self.run_btn.config(state=tk.DISABLED)
        self.log("")
        self.log("🚀 Starting two-stage fault detection...")
        thread = threading.Thread(target=process_data, args=(self.file_path, self.log, self.update_progress))
        thread.daemon = True
        thread.start()

        def check_thread():
            if thread.is_alive():
                self.after(100, check_thread)
            else:
                self.run_btn.config(state=tk.NORMAL)

        check_thread()


# -------------------------- Main Entry --------------------------
if __name__ == "__main__":
    app = FaultDetectApp()
    app.mainloop()