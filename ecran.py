import mss

with mss.mss() as sct:
    print("Ecrans détectés :")
    for i, monitor in enumerate(sct.monitors):
        print(i, monitor)
