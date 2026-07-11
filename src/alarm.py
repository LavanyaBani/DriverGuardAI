import simpleaudio as sa
import threading
import os

alarm_running = False
alarm_thread = None

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ALARM_FILE = os.path.join(
    BASE_DIR,
    "assets",
    "alarm.wav"
)

def alarm_worker():

    wave_obj = sa.WaveObject.from_wave_file(
        ALARM_FILE
    )

    while alarm_running:

        play_obj = wave_obj.play()

        while play_obj.is_playing():

            if not alarm_running:

                play_obj.stop()
                return

       


def start_alarm():

    global alarm_running
    global alarm_thread

    if not alarm_running:

        alarm_running = True

        alarm_thread = threading.Thread(
            target=alarm_worker,
            daemon=True
        )

        alarm_thread.start()


def stop_alarm():

    global alarm_running

    alarm_running = False

