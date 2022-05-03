import torch
import numpy as np
import pandas as pd
import cv2

from torchvision import transforms
from torch.utils.data import DataLoader
from scipy import interpolate, signal

class Track():
    def __init__(self, video_fp = None, model_path = 'best_model.pth') -> None:
        self.res = 320
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        if video_fp is not None:
            self.load_video(video_fp)

    def load_video(self, video_fp) -> None:
        vidcap = cv2.VideoCapture(video_fp)
        self.frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameRate = int(vidcap.get(cv2.CAP_PROP_FPS))
        self.videoLength = self.frameCount / self.frameRate
        inW = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        inH = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res = max(inW, inH)
        self.videoRaw = torch.empty((self.frameCount, 3, res, res))
        
        success = True
        i = 0
        while success:
            success, frame = vidcap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transforms.ToTensor()(frame)
                frame = transforms.CenterCrop((res, res))(frame)
                self.videoRaw[i] = frame
                i += 1
                
    def process_video(self, start: int = 0, stop: int = None) -> None:
        
        if stop is None:
            stop = self.frameCount
        
        assert start < stop
        assert stop <= self.frameCount
        assert start >= 0
        
        self.frameCount = stop - start
        
        self.video = transforms.Resize((self.res, self.res))(self.videoRaw)
        self.video = self.video[start:stop]
        
    def run(self, batch_size=64) -> pd.DataFrame:
        mask = torch.empty((self.frameCount, 2, self.res, self.res), dtype=torch.float32)
        model = torch.load(self.model_path, map_location=self.device)
        dataloader = DataLoader(self.video, batch_size=batch_size)
        filledTo = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                size = batch.shape[0]
                pred = model(batch)

                # add pred to mask
                mask[filledTo:(filledTo + size)] = pred.cpu().detach()
                filledTo += size

            del batch, pred, model
            torch.cuda.empty_cache()
            
        mask = mask.cpu().numpy()
        
        # get contours and fit ellipses
        rows = [None] * self.frameCount
        for idx in range(0, self.frameCount):
            contours_in, _ = cv2.findContours(np.array(mask[idx, 0]*255, dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_out, _ = cv2.findContours(np.array(mask[idx, 1]*255, dtype=np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            largest_in = np.argmax([cv2.contourArea(c) for c in contours_in])
            largest_out = np.argmax([cv2.contourArea(c) for c in contours_out])
            ellipse_in = cv2.fitEllipse(contours_in[largest_in])
            box_in = cv2.boundingRect(contours_in[largest_in])
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
        fits = pd.concat(rows)
        
        heightScale = 0.450/fits[['height_in', 'height_out']].mean(axis=1).quantile(0.5)
        widthScale = 0.450/fits[['width_in', 'width_out']].mean(axis=1).quantile(0.5)
        
        
        # TODO - add in the interpolation
        # td = 0.01
        # tn = np.arange(0, self.videoLength, td)
        
        # splinefn = interpolate.make_interp_spline(fits['t'], fits[['x']], axis=0, k=3)
        # pos_spline = splinefn(tn)
        # pos_smooth = signal.savgol_filter(pos_spline, 30, 3, axis=0, mode='nearest')
        
        
        main = pd.DataFrame()
        main['frame'] = fits['frame']
        main['t'] = fits['t']
        main['x'] = (fits['x_in'] + fits['x_out'])*widthScale / 2
        main['y'] = (fits['y_in'] + fits['y_out'])*heightScale / 2
        main['x'] = main['x'] - main['x'].min()
        main['y'] = main['y'].max() - main['y']
        main['vx'] = np.diff(main['x'], append=np.nan) / np.diff(main['t'], append=np.nan)
        main['vy'] = np.diff(main['y'], append=np.nan) / np.diff(main['t'], append=np.nan)
        main['ax'] = np.diff(main['vx'], append=np.nan) / np.diff(main['t'], append=np.nan)
        main['ay'] = np.diff(main['vy'], append=np.nan) / np.diff(main['t'], append=np.nan)
        
        return main
    
def plot_trajectory(df, out_fp = 'out.png', style = 'seaborn-whitegrid'):
    import matplotlib.pyplot as plt
    
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
    # test
    import time
    start = time.time()
    track = Track(video_fp = 'dev/test/test_input2.mp4', model_path = 'src/bar_tracking/best_model.pth')
    track.process_video()
    df = track.run()
    print(track.videoRaw.shape)
    print(track.video.shape)
    print(df.shape)
    print(df.head())
    plot_trajectory(df)
    end = time.time()
    print(end - start)
    