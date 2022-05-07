import onnxruntime as ort
import numpy as np
import pandas as pd
import cv2
import os
import multiprocessing
from scipy import interpolate, signal

base_path = os.path.dirname(__file__)
model_names = [fn for fn in os.listdir(os.path.dirname(__file__)) if fn.endswith('.onnx')]

class Track():
    def __init__(self, video_fp = None, model_name = 'timm-regnetx_002_model_simplified.onnx', threads = multiprocessing.cpu_count()) -> None:
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.sess_options.intra_op_num_threads = threads
        self.sess_options.inter_op_num_threads = threads
        
        
        self.res = 320
        self.model_path = base_path + '/' + model_name
        self.splinedFit = None
        self.video = None
        
        if video_fp is not None:
            self.load_video(video_fp)

    def load_video(self, video_fp) -> None:
        self.vidcap = cv2.VideoCapture(video_fp)
        self.frameCount = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameRate = int(self.vidcap.get(cv2.CAP_PROP_FPS))
        self.videoLength = self.frameCount / self.frameRate
                
    def process_video(self, start = 0, stop = None, units = ['frames', 'seconds']) -> None:
        if isinstance(units, list):
            units = units[0]
        
        if units == 'seconds':
            start = int(start * self.frameRate)
            stop = int(stop * self.frameRate)
        elif units == 'frames':
            pass
        else:
            raise Exception('Units must be either "frames" or "seconds"')
            
        if stop is None:
            stop = self.frameCount
        
        assert start < stop
        assert stop <= self.frameCount
        assert start >= 0
        
        self.frameCount = stop - start + 1
        self.videoLength = self.frameCount / self.frameRate
        
        self.video = np.empty((self.frameCount, 3, self.res, self.res), dtype=np.float32)
        
        success = True
        i = 0
        idx = 0
        while success and i <= stop:
            success, frame = self.vidcap.read()
            if success and i >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.res, self.res))
                
                frame = frame.astype(np.float32)/255.0
                frame = np.transpose(frame, (2, 0, 1))
                
                self.video[idx] = frame
                idx += 1
            i += 1
        
    def get_splinedFit(self) -> pd.DataFrame:
        if self.splinedFit is None:
            self.run()
        
        return self.splinedFit
        
    def run(self, td = 0.025) -> pd.DataFrame:
        if self.video is None:
            self.process_video()
        
        mask = np.empty((self.frameCount, 2, self.res, self.res), dtype=np.float32)
        session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], session_options=self.sess_options)
        input_name = session.get_inputs()[0].name
        
        # get contours and fit ellipses
        rows = [None] * self.frameCount
        for idx in range(0, self.frameCount):
            mask[idx] = session.run(None, {input_name: self.video[idx].reshape(1, 3, self.res, self.res)})[0]
            try:
                contours_in, _ = cv2.findContours(np.array(mask[idx, 0]*255, dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                largest_in = np.argmax([cv2.contourArea(c) for c in contours_in])
                ellipse_in = cv2.fitEllipse(contours_in[largest_in])
                box_in = cv2.boundingRect(contours_in[largest_in])
                contours_out, _ = cv2.findContours(np.array(mask[idx, 1]*255, dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                largest_out = np.argmax([cv2.contourArea(c) for c in contours_out])
                ellipse_out = cv2.fitEllipse(contours_out[largest_out])
                box_out = cv2.boundingRect(contours_out[largest_out])
                
                rows[idx] = pd.DataFrame({
                    'frame': idx,
                    't': idx / self.frameRate,
                    'x_in': ellipse_in[0][0],
                    'x_out': ellipse_out[0][0],
                    'y_in': ellipse_in[0][1],
                    'y_out': ellipse_out[0][1],
                    'height_in': box_in[3],
                    'height_out': box_out[3],
                    'width_in': box_in[2],
                    'width_out': box_out[2],
                }, index=[idx])
            except:
                print('Error on frame {}'.format(idx))
                rows[idx] = pd.DataFrame({
                    'frame': idx,
                    't': idx / self.frameRate
                }, index=[idx])
                
            
        fits = pd.concat(rows)
        
        # interpolate to standard time resolution
        tn = np.arange(0, self.videoLength, td)
        
        t = fits['t'].to_numpy()
        pos = fits.iloc[:, 2:].to_numpy()
        
        splinefn = interpolate.make_interp_spline(t, pos, axis=0, k=3, check_finite=False)
        splinedFit = pd.DataFrame(splinefn(tn), columns=fits.columns[2:])
        
        heightScale = 0.450/splinedFit[['height_in', 'height_out']].mean(axis=1).quantile(0.5)
        widthScale = 0.450/splinedFit[['width_in', 'width_out']].mean(axis=1).quantile(0.5)
        
        # scale to meters
        splinedFit[['height_in', 'height_out', 'y_in', 'y_out']] *= heightScale
        splinedFit[['width_in', 'width_out', 'x_in', 'x_out']] *= widthScale
        
        # apply savgol filter to each column
        for col in splinedFit.columns:
            splinedFit[col] = signal.savgol_filter(splinedFit[col].to_numpy(), 25, 3)
        
        splinedFit.insert(0, 't', tn)
        
        splinedFit['x'] = (splinedFit['x_in'] + splinedFit['x_out']) / 2
        splinedFit['y'] = (splinedFit['y_in'] + splinedFit['y_out']) / 2
        splinedFit['x'] = splinedFit['x'] - splinedFit['x'].min()
        splinedFit['y'] = splinedFit['y'].max() - splinedFit['y']
        splinedFit['vx'] = np.diff(splinedFit['x'], append=np.nan) / td
        splinedFit['vy'] = np.diff(splinedFit['y'], append=np.nan) / td
        splinedFit['ax'] = np.diff(splinedFit['vx'], append=np.nan) / td
        splinedFit['ay'] = np.diff(splinedFit['vy'], append=np.nan) / td
        
        self.splinedFit = splinedFit
        return splinedFit
    
def plot_trajectory(track: Track, out_fp = 'out.png', style = 'seaborn-whitegrid'):
    import matplotlib.pyplot as plt
    
    df = track.get_splinedFit()
    
    colors = ['green', 'red', '#0099ff']
    lwd = 3
    
    plt.style.use(style)
    plt.figure(figsize=(15, 7), facecolor='white')
    ax1 = plt.subplot(131)
    ax1.set_aspect('equal')
    ax1.plot(df.x, df.y, color=colors[0], linewidth=lwd)
    ax1.set_title('Bar Path')
    ax1.set_xbound(-0.2, 0.6)

    ax2 = plt.subplot(332)
    ax2.plot(df.t, df.y, color = colors[1], linewidth=lwd)
    ax2.set_title('y Attributes vs Time')

    ax3 = plt.subplot(333)
    ax3.plot(df.t, df.x, color = colors[2], linewidth=lwd)
    ax3.set_title('x Attributes vs Time')

    ax4 = plt.subplot(335)
    ax4.plot(df.t, df.vy, color = colors[1], linewidth=lwd)

    ax5 = plt.subplot(336)
    ax5.plot(df.t, df.vx, color = colors[2], linewidth=lwd)

    ax6 = plt.subplot(338)
    ax6.plot(df.t, df.ay, color = colors[1], linewidth=lwd)

    ax7 = plt.subplot(339)
    ax7.plot(df.t, df.ax, color = colors[2], linewidth=lwd)

    ax2.set_ylabel('position (m)')
    ax4.set_ylabel('velocity (m/s)')
    ax6.set_ylabel('acceleration (m/s^2)')

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')

    ax6.set_xlabel('Time (s)')
    ax7.set_xlabel('Time (s)')
    
    for ax in [ax4, ax5, ax6, ax7]:
        ax.axhline(y = 0, color = 'gray', linestyle = 'dashed')
        
    plt.savefig(out_fp, transparent=False, dpi = 300, bbox_inches='tight', facecolor='white')
    plt.close()
    
if __name__ == '__main__':
    import time
    start = time.time()
    track = Track(video_fp = 'dev/test/test_input2.mp4')
    track.process_video(0, 2, units = 'seconds')
    plot_trajectory(track, out_fp = '.test.png')
    end = time.time()
    print(end - start, 's elapsed')
    print(((end-start)/track.frameCount)*1000, 'ms/frame\n')
    print(track.get_splinedFit())
