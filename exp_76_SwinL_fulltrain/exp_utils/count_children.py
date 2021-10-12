import os
import psutil
import time

def pause_for_children():
    cnt, spin = 0, ["-", "\\", "|", "/", "-", "\\", "|", "/"]
    main_proc = psutil.Process(os.getpid())
    print(f"main process has {len(main_proc.children(recursive=True))} children")
    while len(main_proc.children(recursive=True)) > 1:
        print(f"killing workers ::: {spin[cnt%len(spin)]}",end='\r')
        cnt += 1
        time.sleep(0.1)

    print('done!')