import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    import por_trigger
    import deltae_scoring
    import grv_scoring
