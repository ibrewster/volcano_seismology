import time
from VolcSeismo.process import run

if __name__ == "__main__":
    t1 = time.time()
    run()
    print("Data run complete in", (time.time()-t1)/60, "minutes")
