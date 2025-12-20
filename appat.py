import time

import pyautogui


INTERVAL = 1  # secondes

print("[INFO] Envoi de NUMPAD '-' toutes les 90 secondes.")
print("[INFO] Appuie sur CTRL+C dans le terminal pour arrêter.")

# petite pause pour laisser le temps de focus la fenêtre cible
time.sleep(3)


def send_keyso_from_text(text):
    for c in text:
        if c.isdigit():
            key_numpad = f"num{c}"
            print(f"Sending digit (numpad): {key_numpad}")

        else:
            key = c.lower()
            print(f"Sending letter: {key}")
            pyautogui.press(key)

        time.sleep(0.03)

try:
    while True:
        # Touche '-' du pavé numérique
        send_keyso_from_text("O")

        print("[SEND] NUMPAD '-' envoyé.")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n[INFO] Arrêt du script.")
