from ghost_engine import GhostMouseEngine
import time

print("Testing GhostEngine...")
engine = GhostMouseEngine()
engine.start(use_face=False)

print("Engine started. Running for 10 seconds...")
time.sleep(10)
print("Stopping engine...")
engine.stop()
print("Done.")
