
import sys
import subprocess
import numpy as np
import time
import os
import requests
import select
import fcntl
from urllib.parse import urljoin
from scipy.fft import rfft, rfftfreq
from dotenv import load_dotenv
from datetime import datetime


# =============================================================================
# Глобальные параметры
# =============================================================================
# --- Веб-хук ---
WEBHOOK_URL = "http://homeassistant.local:8123/api/webhook/-XZTiYxnPRIY288ryyIXNGQsS"

# --- Параметры детектора ---
TARGET_FREQUENCIES = [1020, 3040]
FREQUENCY_TOLERANCE = 40
ENERGY_THRESHOLD_MULTIPLIER = 3
NOISE_FLOOR_HZ = 800

# --- Параметры потока ---
SAMPLE_RATE = 8000
CHUNK_DURATION_S = 0.064
HOP_DURATION_S = 0.032

# --- Параметры последовательности ---
TARGET_BEEP_COUNT = 4
SEQUENCE_TIMEOUT_S = 4.0
MIN_BEEP_INTERVAL_S = 0.1

LOG_INTERVAL_S = 300.0

# --- Вычисляемые константы ---
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION_S)
MIN_INTERVAL_SAMPLES = int(SAMPLE_RATE * MIN_BEEP_INTERVAL_S)
HAMMING_WINDOW = np.hamming(CHUNK_SAMPLES)

# =============================================================================
# Класс для получения RTSP-ссылки (из recorder.py)
# =============================================================================
class UfanetCam:
    """
    Класс для аутентификации и получения RTSP-ссылки на поток с камер Ufanet.
    """
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _get_access_token(self):
        print("Авторизация в ufanet...")
        url = "https://dom.ufanet.ru/api/v1/auth/auth_by_contract/"
        data = {"contract": self.username, "password": self.password}
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()["token"]["access"]
        except requests.RequestException as e:
            print(f"Ошибка сетевого запроса: {e}")
            return None

    def _get_cam_server_url(self, access_token):
        print("Получение URL сервиса камер...")
        url = "https://dom.ufanet.ru/api/v0/contract/"
        self.session.headers.update({"Authorization": f"JWT {access_token}"})
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()[0]["isp_org"]["cams_server"]["url"]
        except (requests.RequestException, IndexError, KeyError) as e:
            print(f"Ошибка сетевого запроса: {e}")
            return None

    def _get_ucams_token(self, cam_server_url, access_token):
        print("Авторизация в сервисе камер...")
        url = urljoin(cam_server_url, "/api/v0/auth/?ttl=20800")
        self.session.headers.update({"Authorization": f"JWT {access_token}"})
        try:
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()["token"]
        except requests.RequestException as e:
            print(f"Ошибка сетевого запроса: {e}")
            return None

    def _get_camera_info(self, cam_server_url, ucams_token):
        print("Получение информации о камере...")
        url = urljoin(cam_server_url, "/api/v0/cameras/my/")
        self.session.headers.update({"Authorization": f"Bearer {ucams_token}"})
        data = {
            "order_by": "addr_asc",
            "fields": ["number", "server", "token_l"],
            "token_l_ttl": 86400,
            "page": 1,
            "page_size": 1,
        }
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()["results"][0]
        except (requests.RequestException, IndexError, KeyError) as e:
            print(f"Ошибка сетевого запроса: {e}")
            return None

    def get_rtsp_url(self):
        """
        Выполняет все шаги аутентификации и возвращает готовую RTSP-ссылку.
        При любой ошибке возвращает None.
        """
        access_token = self._get_access_token()
        if access_token is None:
            return None

        cam_server_url = self._get_cam_server_url(access_token)
        if cam_server_url is None:
            return None

        ucams_token = self._get_ucams_token(cam_server_url, access_token)
        if ucams_token is None:
            return None

        cam_info = self._get_camera_info(cam_server_url, ucams_token)
        if cam_info is None:
            return None

        cam_id = cam_info["number"]
        token_l = cam_info["token_l"]
        domain = cam_info["server"]["domain"]

        rtsp_link = f"rtsp://{domain}/{cam_id}?token={token_l}&tracks=v1a1"
        return rtsp_link

# =============================================================================
# Класс детектора (из detector4.py)
# =============================================================================
class BeepDetector:
    def __init__(self):
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.total_samples_processed = 0
        self.last_detection_sample = -MIN_INTERVAL_SAMPLES
        self.frequencies_fft = rfftfreq(CHUNK_SAMPLES, d=1.0 / SAMPLE_RATE)
        self.noise_floor_index = np.searchsorted(self.frequencies_fft, NOISE_FLOOR_HZ)
        self.last_log_timestamp = 0

    def process_new_data(self, new_audio_bytes: bytes):
        new_samples = np.frombuffer(new_audio_bytes, dtype=np.int16).astype(np.float32)
        self.audio_buffer = np.concatenate((self.audio_buffer, new_samples))
        detected_timestamps = []
        while len(self.audio_buffer) >= CHUNK_SAMPLES:
            chunk = self.audio_buffer[:CHUNK_SAMPLES]
            timestamp_s = self.total_samples_processed / SAMPLE_RATE
            if self._is_beep(chunk):
                detected_timestamps.append(timestamp_s)
            self.audio_buffer = self.audio_buffer[HOP_SAMPLES:]
            self.total_samples_processed += HOP_SAMPLES
        return detected_timestamps

    def _is_beep(self, chunk: np.ndarray) -> bool:
        if self.total_samples_processed < self.last_detection_sample + MIN_INTERVAL_SAMPLES:
            return False
        chunk *= HAMMING_WINDOW
        spectrum = np.abs(rfft(chunk))
        mean_energy = np.mean(spectrum[self.noise_floor_index:])
        if mean_energy < 1e-6: mean_energy = 1e-6
        ratios = {}
        for target_freq in TARGET_FREQUENCIES:
            freq_band_indices = np.where(
                (self.frequencies_fft >= target_freq - FREQUENCY_TOLERANCE) &
                (self.frequencies_fft <= target_freq + FREQUENCY_TOLERANCE)
            )[0]
            if freq_band_indices.size == 0: return False
            peak_energy_in_band = np.max(spectrum[freq_band_indices])
            ratios[target_freq] = float(peak_energy_in_band / mean_energy)
            if datetime.now().timestamp() - self.last_log_timestamp > LOG_INTERVAL_S:
                print(f"{datetime.now()}: {ratios}")
                self.last_log_timestamp = datetime.now().timestamp()
            if peak_energy_in_band < mean_energy * ENERGY_THRESHOLD_MULTIPLIER:
                return False
        self.last_detection_sample = self.total_samples_processed
        # print(ratios)
        return True

# =============================================================================
# Функции для веб-хука и запуска
# =============================================================================
def trigger_webhook(is_testing: bool):
    """Отправляет POST-запрос на заданный URL веб-хука."""
    print("Отправка веб-хука...")
    if is_testing:
        return
    try:
        response = requests.post(WEBHOOK_URL, timeout=5)
        if 200 <= response.status_code < 300:
            print("Веб-хук успешно отправлен!")
        else:
            print(f"Ошибка веб-хука: Статус {response.status_code}, Ответ: {response.text}")
    except requests.RequestException as e:
        print(f"Ошибка при отправке веб-хука: {e}")

def run_live_detector(source: str, is_file: bool):
    """Запускает FFmpeg, анализирует поток и вызывает веб-хук при обнаружении."""
    print(f"Запуск FFmpeg для источника: {source}")
    command = [
        'ffmpeg', '-rtsp_transport', 'tcp', '-i', source,
        '-hide_banner',
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLE_RATE), '-ac', '1', '-f', 's16le', '-'
    ] if not is_file else [
        'ffmpeg', '-i', source,
        '-hide_banner',
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLE_RATE), '-ac', '1', '-f', 's16le', '-'
    ]

    ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # --- Настройка неблокирующего чтения --- 
    stdout_fd = ffmpeg_process.stdout.fileno()
    fcntl.fcntl(stdout_fd, fcntl.F_SETFL, os.O_NONBLOCK)
    # ----------------------------------------

    detector = BeepDetector()
    beep_count = 0
    last_beep_timestamp = -SEQUENCE_TIMEOUT_S

    ready_to_read, _, _ = select.select([ffmpeg_process.stdout], [], [], 10.0)
    if not ready_to_read:
        print("Поток не начался за 10 секунд. Пробую ещё раз.")
        ffmpeg_process.kill() # Убиваем зависший процесс
        return
    print("Начинаю слушать поток...")
    while True:
        # Проверяем, жив ли еще процесс FFmpeg
        if ffmpeg_process.poll() is not None:
            print("Процесс FFmpeg завершился.")
            break

        # Ждем данных в stdout с таймаутом
        ready_to_read, _, _ = select.select([ffmpeg_process.stdout], [], [], 5.0)

        if not ready_to_read:
            # Если данных нет, считаем, что поток завис
            print("Нет данных от FFmpeg. Поток, вероятно, завис.")
            ffmpeg_process.kill() # Убиваем зависший процесс
            break

        # Читаем все доступные данные, чтобы не отставать
        raw_audio = ffmpeg_process.stdout.read()

        detected_timestamps = detector.process_new_data(raw_audio)

        for ts in detected_timestamps:
            if is_file:
                print(f"{datetime.now()} → Обнаружен сигнал на {ts}", end='')
            else:
                print(f"{datetime.now()} → Обнаружен сигнал", end='')
            if ts - last_beep_timestamp > SEQUENCE_TIMEOUT_S:
                beep_count = 1
            else:
                beep_count += 1
            print(f" (Нажатие #{beep_count} в серии)")
            last_beep_timestamp = ts

            if beep_count == TARGET_BEEP_COUNT:
                trigger_webhook(is_file)
                beep_count = 0

    if not is_file:
        print("Проверяю stderr...")
        _, stderr = ffmpeg_process.communicate()
        error_message = stderr.decode(errors='ignore').strip()
        if error_message:
            print("--- Ошибка FFmpeg ---")
            print(error_message)
            print("---------------------")

# =============================================================================
# Точка входа
# =============================================================================
if __name__ == '__main__':
    """
    Использование
    python main.py — спарсить из .env UFA_USER и UFA_PASS, начать прослушивать реальный поток
    python main.py media_file — для тестов: использовать media_file в качестве источника, не вызывать хук
    """
    print("Загрузка учетных данных из .env файла...")
    load_dotenv()
    username = os.getenv("UFA_USER")
    password = os.getenv("UFA_PASS")

    if len(sys.argv) == 2:
        # testing branch
        run_live_detector(sys.argv[1], True)
        sys.exit(0)

    if not username or not password:
        print("Ошибка: Убедитесь, что UFA_USER и UFA_PASS заданы в .env файле.")
        sys.exit(1)

    cam_auth = UfanetCam(username, password)

    while True:
        print("-"*60)
        rtsp_url = cam_auth.get_rtsp_url()
        if rtsp_url:
            run_live_detector(rtsp_url, False)
        else:
            print("Не удалось получить RTSP ссылку. Повторная попытка через 2 секунды...")
            time.sleep(2)
            continue

        print("Перезапуск процесса...")
