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

# --- Общие параметры обработки аудио ---
SAMPLE_RATE = 8000
CHUNK_DURATION_S = 0.064
HOP_DURATION_S = 0.032
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION_S)
HAMMING_WINDOW = np.hamming(CHUNK_SAMPLES)
FREQUENCIES_FFT = rfftfreq(CHUNK_SAMPLES, d=1.0 / SAMPLE_RATE)

# --- Профили для распознавания разных звуков ---
SOUND_PROFILES = {
    "BUTTON_PRESS": {
        "name": "Нажатие кнопки",
        "env_var": "DETECT_BUTTONS",
        "webhook_url_env_var": "WEBHOOK_URL_BUTTONS",
        "params": {
            "target_frequencies": [1020, 3040],
            "frequency_tolerance": 40,
            "energy_threshold_multiplier": 3,
            "noise_floor_hz": 800,
            "min_interval_s": 0.1,
        },
        "sequence": {
            "target_count": 4,
            "timeout_s": 4.0,
        }
    },
    "CALL_TONE": {
        "name": "Звонок",
        "env_var": "DETECT_CALL",
        "webhook_url_env_var": "WEBHOOK_URL_CALL",
        "params": {
            "target_frequencies": [700],
            "frequency_tolerance": 40,
            "energy_threshold_multiplier": 25,
            "noise_floor_hz": 600,
            "min_interval_s": 0.13,
        },
        "sequence": {
            "target_count": 4,
            "timeout_s": 5.0,
        }
    }
}


class SingleSoundDetector:
    """Распознает один конкретный тип звука на основе его профиля."""
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params
        self.noise_floor_index = np.searchsorted(FREQUENCIES_FFT, self.params["noise_floor_hz"])
        self.min_interval_samples = int(SAMPLE_RATE * self.params["min_interval_s"])
        self.last_detection_sample = -self.min_interval_samples

    def check(self, chunk: np.ndarray, total_samples_processed: int) -> bool:
        """Проверяет, содержит ли аудио-чанк заданный звук."""
        if total_samples_processed < self.last_detection_sample + self.min_interval_samples:
            return False

        chunk_with_window = chunk * HAMMING_WINDOW
        spectrum = np.abs(rfft(chunk_with_window))
        mean_energy = np.mean(spectrum[self.noise_floor_index:])
        if mean_energy < 1e-6: mean_energy = 1e-6

        for target_freq in self.params["target_frequencies"]:
            freq_band_indices = np.where(
                (FREQUENCIES_FFT >= target_freq - self.params["frequency_tolerance"]) &
                (FREQUENCIES_FFT <= target_freq + self.params["frequency_tolerance"])
            )[0]
            if freq_band_indices.size == 0:
                return False
            peak_energy_in_band = np.max(spectrum[freq_band_indices])

            if peak_energy_in_band < mean_energy * self.params["energy_threshold_multiplier"]:
                return False

        self.last_detection_sample = total_samples_processed
        return True


class SequenceTracker:
    """Отслеживает последовательность обнаруженных звуков и вызывает веб-хук."""
    def __init__(self, name: str, target_count: int, timeout_s: float, webhook_url: str):
        self.name = name
        self.target_count = target_count
        self.timeout_s = timeout_s
        self.webhook_url = webhook_url
        self.count = 0
        self.last_timestamp = -timeout_s

    def add_detection(self, timestamp: float, is_testing: bool):
        """Регистрирует новое обнаружение и проверяет, завершена ли серия."""
        if is_testing:
            print(f"{datetime.now()} → Обнаружен сигнал '{self.name}' на {timestamp}", end='')
        else:
            print(f"{datetime.now()} → Обнаружен сигнал '{self.name}'", end='')

        if timestamp - self.last_timestamp > self.timeout_s:
            self.count = 1
        else:
            self.count += 1
        print(f" (#{self.count} в серии)")
        self.last_timestamp = timestamp

        if self.count >= self.target_count:
            self._trigger_webhook(is_testing)
            self.count = 0

    def _trigger_webhook(self, is_testing: bool):
        """Отправляет POST-запрос на заданный URL веб-хука."""
        if not self.webhook_url:
            print(f"URL веб-хука для '{self.name}' не задан, пропуск.")
            return
        print(f"Отправка веб-хука для '{self.name}' на {self.webhook_url}...")
        if is_testing:
            print("(Тестовый режим, реальная отправка отменена)")
            return
        try:
            response = requests.post(self.webhook_url, timeout=5)
            if 200 <= response.status_code < 300:
                print("Веб-хук успешно отправлен")
            else:
                print(f"Ошибка веб-хука: Статус {response.status_code}, Ответ: {response.text}")
        except requests.RequestException as e:
            print(f"Ошибка при отправке веб-хука: {e}")


class MultiDetectorProcessor:
    """
    Главный обработчик аудиопотока. Управляет всеми активными детекторами
    и трекерами последовательностей.
    """
    def __init__(self, enabled_profiles: list):
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.total_samples_processed = 0
        self.detectors = [SingleSoundDetector(p["name"], p["params"]) for p in enabled_profiles]
        self.trackers = [SequenceTracker(p["name"], p["sequence"]["target_count"], p["sequence"]["timeout_s"], p.get("webhook_url")) for p in enabled_profiles]

    def process_new_data(self, new_audio_bytes: bytes, is_testing: bool):
        """Обрабатывает новый кусок аудио, проверяя его всеми детекторами."""
        new_samples = np.frombuffer(new_audio_bytes, dtype=np.int16).astype(np.float32)
        self.audio_buffer = np.concatenate((self.audio_buffer, new_samples))

        while len(self.audio_buffer) >= CHUNK_SAMPLES:
            chunk = self.audio_buffer[:CHUNK_SAMPLES]

            for i, detector in enumerate(self.detectors):
                if detector.check(chunk, self.total_samples_processed):
                    timestamp_s = self.total_samples_processed / SAMPLE_RATE
                    self.trackers[i].add_detection(timestamp_s, is_testing)

            self.audio_buffer = self.audio_buffer[HOP_SAMPLES:]
            self.total_samples_processed += HOP_SAMPLES


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

def run_live_detector(source: str, is_file: bool, processor: MultiDetectorProcessor):
    """Запускает FFmpeg, передает аудиопоток процессору MultiDetectorProcessor."""
    print(f"Запуск FFmpeg для источника: {source}")
    command = [
        'ffmpeg', '-rtsp_transport', 'tcp', '-i', source,
        '-hide_banner', '-loglevel', 'error',
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLE_RATE), '-ac', '1', '-f', 's16le', '-'
    ] if not is_file else [
        'ffmpeg', '-i', source,
        '-hide_banner', '-loglevel', 'error',
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLE_RATE), '-ac', '1', '-f', 's16le', '-'
    ]

    ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Настройка неблокирующего чтения
    stdout_fd = ffmpeg_process.stdout.fileno()
    fcntl.fcntl(stdout_fd, fcntl.F_SETFL, os.O_NONBLOCK)

    ready_to_read, _, _ = select.select([ffmpeg_process.stdout], [], [], 10.0)
    if not ready_to_read:
        print("Поток не начался за 10 секунд")
        ffmpeg_process.kill() # Убиваем зависший процесс
        return
    print("Начинаю слушать поток...")
    last_cycle = False
    while True:
        if last_cycle:
            print("Процесс FFmpeg завершился")
            break

        # Проверяем, жив ли еще процесс FFmpeg
        if ffmpeg_process.poll() is not None:
            # Нужно дочитать данные в stdout
            last_cycle = True

        # Ждем данных в stdout с таймаутом
        ready_to_read, _, _ = select.select([ffmpeg_process.stdout], [], [], 5.0)

        if not ready_to_read:
            # Если данных нет, считаем, что поток завис
            print("Нет данных от FFmpeg. Поток, вероятно, завис.")
            ffmpeg_process.kill() # Убиваем зависший процесс
            break

        # Читаем все доступные данные, чтобы не отставать
        raw_audio = ffmpeg_process.stdout.read()

        if raw_audio:
            processor.process_new_data(raw_audio, is_file)

    # Выводим ошибки FFmpeg, если они были
    if not is_file:
        _, stderr = ffmpeg_process.communicate()
        error_message = stderr.decode(errors='ignore').strip()
        if error_message:
            print("--- Ошибка FFmpeg ---")
            print(error_message)
            print("---------------------")


def get_configured_profiles() -> list:
    enabled_profiles = []
    for key, profile in SOUND_PROFILES.items():
        if os.getenv(profile["env_var"], "false").lower() in ("true", "1"):
            # Добавляем URL веб-хука из .env в словарь профиля
            profile["webhook_url"] = os.getenv(profile["webhook_url_env_var"])
            enabled_profiles.append(profile)
    return enabled_profiles


if __name__ == '__main__':
    """
    Использование:
    - Запуск с реального потока:
      python main.py
      (требует .env с UFA_USER, UFA_PASS и активными детекторами)

    - Запуск для теста на файле:
      python main.py <путь_к_файлу>
    """
    load_dotenv()
    active_profiles = get_configured_profiles()
    if not active_profiles:
        print("ВНИМАНИЕ: Ни один детектор не включен. Проверьте .env файл.")
        print("Чтобы включить детектор, установите соответствующую переменную в 'true':")
        for _, profile in SOUND_PROFILES.items():
            print(f"- {profile['env_var']} (для '{profile['name']}')")
        sys.exit(1)


    print("Активные детекторы:")
    for p in active_profiles:
        print(f"- {p['name']}")
    print()
    # Главный процессор, который будет делать всю работу
    processor = MultiDetectorProcessor(active_profiles)

    # Ветка для тестирования на локальном файле
    if len(sys.argv) == 2:
        media_file = sys.argv[1]
        print(f"--- РЕЖИМ ТЕСТИРОВАНИЯ ---")
        if not os.path.exists(media_file):
            print(f"Ошибка: Файл не найден: {media_file}")
            sys.exit(1)
        run_live_detector(media_file, True, processor)
        sys.exit(0)

    # Основная ветка для работы с реальным потоком
    print("--- РЕЖИМ РАБОТЫ С RTSP-ПОТОКОМ ---")
    username = os.getenv("UFA_USER")
    password = os.getenv("UFA_PASS")
    if not username or not password:
        print("Ошибка: Убедитесь, что UFA_USER и UFA_PASS заданы в .env файле.")
        sys.exit(1)

    cam_auth = UfanetCam(username, password)

    while True:
        print("-" * 60)
        rtsp_url = cam_auth.get_rtsp_url()
        if rtsp_url:
            # Запускаем детектор с уже созданным и настроенным процессором
            run_live_detector(rtsp_url, False, processor)
        else:
            print("Не удалось получить RTSP ссылку. Повторная попытка через 2 секунды...")
            time.sleep(2)
            continue

        print("Перезапуск процесса...")
