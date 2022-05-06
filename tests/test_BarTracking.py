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