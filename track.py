import torch
import numpy as np
import pandas as pd
import cv2

from skimage.measure import centroid, find_contours
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy import interpolate, signal

class Track():
    def __init__(self, video_fp = None) -> None:
        self.res = 320
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
                frame = cv2.resize(frame, (res, res))
                frame = transforms.ToTensor()(frame)
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
        mask = torch.empty((self.frameCount, 3, self.res, self.res), dtype=torch.float32)
        model = torch.load(f'./best_model.pth', map_location=self.device)
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
        
        position = np.empty((self.frameCount, 2, 2))
        box = np.empty((self.frameCount, 2, 4))

        idx = 0
        for frame in mask:
            position[idx, 0, :] = centroid(frame[0])[::-1]
            position[idx, 1, :] = centroid(frame[1])[::-1]
            #position[idx, 2, :] = centroid(frame[2])[::-1]

            contour = find_contours(frame[0])[0], find_contours(frame[1])[0]

            box[idx] = np.stack((
                np.concatenate((contour[0].min(axis=0), contour[0].max(axis=0))),
                np.concatenate((contour[1].min(axis=0), contour[1].max(axis=0))),
                ))

            idx += 1

        # pixel coordinates to real world meters
        position = position/self.res
        
        # invert y axis
        position[:, :, 1] = 1 - position[:, :, 1]
        
        position = signal.savgol_filter(position, 5, 2, axis=0, mode='nearest')
        
        dim = np.stack((box[:, :, 3] - box[:, :, 1], box[:, :, 2] - box[:, :, 0]), axis=-1)
        dim = signal.savgol_filter(dim, 50, 1, axis=0, mode='nearest')

        self.coef = 0.450/np.quantile(dim.mean(axis=1), 0.50, axis=0)/self.res
        position = np.multiply(position, self.coef)
        
        # average position of each side
        position = position.mean(axis=1)
        
        #position = position - position.mean(axis=1).min(axis=0)
        
        length = self.frameCount/self.frameRate
        t = np.linspace(0, length, self.frameCount)
        td = 0.01
        tn = np.arange(0, length, td)

        # create spline, interpolate, and smooth
        splinefn = interpolate.make_interp_spline(t, position, axis=0, k=3)
        pos_spline = splinefn(tn)
        pos_smooth = signal.savgol_filter(pos_spline, 30, 3, axis=0, mode='nearest')

        # velocity and acceleration
        vel = np.diff(pos_smooth, axis=0)/td
        accel = np.diff(vel, axis=0)/td

        vel_smooth = signal.savgol_filter(vel, 50, 2, axis=0, mode='interp')
        accel_smooth = signal.savgol_filter(accel, 75, 1, axis=0, mode='constant')
        
        data = np.concatenate((tn.reshape(-1, 1)[1:-1], pos_smooth[1:-1], vel_smooth[:-1], accel_smooth), axis=1)

        # make dataframe
        self.dataframe = pd.DataFrame(data, columns=['t', 'x', 'y', 'vx', 'vy', 'ax', 'ay'])
        return self.dataframe

if __name__ == '__main__':
    # test
    track = Track(video_fp = './test/test_input2.mp4')
    track.process_video(start = 0, stop = 60)
    df = track.run()
    print(track.videoRaw.shape)
    print(track.video.shape)
    print(df.shape)
    print(df.head())
    