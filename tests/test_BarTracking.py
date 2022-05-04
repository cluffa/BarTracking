from BarTracking.track import *
import os

dir = os.path.dirname(os.path.abspath(__file__))

def test_load():
    try:
        Track(video_fp = dir + '/test.mp4')
        assert True
    except Exception as e:
        print(e)
        assert False
    
def test_process():
    try:
        t = Track(video_fp = dir + '/test.mp4')
        t.process_video()
        assert True
    except Exception as e:
        print(e)
        assert False
    
def test_run():
    try:
        t = Track(video_fp = dir + '/test.mp4')
        t.run()
        assert True
    except Exception as e:
        print(e)
        assert False
    
def test_graph():
    try:
        t = Track(video_fp = dir + '/test.mp4')
        plot_trajectory(t, dir + '/test_output.png')
        assert True
    except Exception as e:
        print(e)
        assert False