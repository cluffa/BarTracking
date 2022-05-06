import os

dir = os.path.dirname(os.path.abspath(__file__))
    
from BarTracking.track import *

t = Track(video_fp = dir + '/test.mp4')
    
def test_process():
    try:
        t.process_video()
        assert True
    except Exception as e:
        print(e)
        assert False
    
def test_run():
    try:
        t.run()
        assert True
    except Exception as e:
        print(e)
        assert False
    
def test_graph():
    try:
        plot_trajectory(t, dir + '/test_output.png')
        assert True
    except Exception as e:
        print(e)
        assert False
        
# full test
def test_full():
    import time
    start = time.time()
    track = Track(video_fp = 'dev/test/test_input2.mp4')
    track.process_video(0, 2, units = 'seconds')
    plot_trajectory(track, out_fp = '.test.png')
    end = time.time()
    print(end - start, 's elapsed')
    print(((end-start)/track.frameCount)*1000, 'ms/frame\n')
    print(track.get_splinedFit())
    
if __name__ == '__main__':
    test_full()